#!/usr/bin/env python
"""
Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and Technology
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

# basic libs
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import Dataset, DataLoader
import git

import time
# networks and dataset
from networks.networks import  RNN, RNN2, TCN
from dataset.Dataset import dbioDataset

# yaml configs
import yaml

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_receptive(kernel_size, levels, dilation_exponential_base): 
    return sum([dilation_exponential_base**(l-1)*(kernel_size-1) for l in range(levels, 0, -1)]) + 1

def input_data_gen(data, network_config, train=False):
    input_data_mask = network_config["network"]["input_data_mask"]
    
    if network_config["dataset"]["gravity_correction"]:
        input_data = torch.cat((input_data_mask[0]*data["alphasense_imu_ds_gc_normed"][:, :,:6],
                                input_data_mask[1]*data["motor_command_normed"][:,:,:8], 
                                input_data_mask[2]*data["orientation_data_Rmat_frag_flat"]), dim=2).cuda()
    else:
        input_data = torch.cat((input_data_mask[0]*data["alphasense_imu_ds_normed"][:, :,:6],
                                input_data_mask[1]*data["motor_command_normed"][:,:,:8],
                                input_data_mask[2]*data["battery_voltage"][:,:,:2]), dim=2).cuda()
    
    return input_data[:, :, :]

if __name__ == '__main__':
    training_start_time = time.time()
    network_config = yaml.load(open("/home/sakura/rosworkspaces/catkin_ws/src/deepvl/scripts/configs/training_config.yaml"), Loader=yaml.FullLoader)
    
    dataset_dir = network_config["dataset"]["dataset_dir"]
    train_dataset_dir = os.path.join(dataset_dir,"train_full")
    # train_dataset_dir = os.path.join(dataset_dir,"test")
    test_dataset_dir = os.path.join(dataset_dir,"test")

    train_dataset = dbioDataset(train_dataset_dir, dataset_config = network_config["dataset"], train=True)
    test_dataset = dbioDataset(test_dataset_dir, dataset_config = network_config["dataset"], train=False)

    print("Length of train and test dataset: ", len(train_dataset), len(test_dataset))

    train_slice = DataLoader(train_dataset, batch_size=network_config["training"]["batch_size"]["train"], shuffle=True, num_workers=2)
    test_slice = DataLoader(test_dataset, batch_size=network_config["training"]["batch_size"]["test"], shuffle=True, num_workers=2)
    
    rnn_list = []

    ensemble_size = network_config["network"]["ensemble_size"]
    
    # LSTM
    if  network_config["network"]["rnn_type"] == "tcn":
        rnn = TCN(network_config["network"]["input_size"],
                  network_config["network"]["output_size"],
                  network_config["network"]["layers"],
                  network_config["network"]["kernel_size"],
                  network_config["network"]["dropout"]).cuda()
        
        print("receptive field of TCN: ", get_receptive(network_config["network"]["kernel_size"], len(network_config["network"]["layers"]), 2))
        
    elif network_config["network"]["rnn_type"] == "gru":
        for i in range(ensemble_size):
            rnn_list.append(RNN(network_config["network"]["input_size"], 
                                network_config["network"]["output_size"], 
                                network_config["network"]["layers"], 
                                3, 
                                network_config["network"]["dropout"],
                                skip_connection=network_config["network"]["skip_connection"]).cuda())
    
    # # # Load params    
    if network_config["utils"]["previous_checkpt_path"] != "":
        for i in range(ensemble_size):
            rnn_list[i].load_state_dict(torch.load(network_config["utils"]["previous_checkpt_path"]))

    # count number of params in the network
    print("Total Param count: ", count_parameters(rnn_list[0]))

    # optimizer
    optimizer_list = []
    scheduler_list = []
    for i in range(ensemble_size):
        optimizer_list.append(torch.optim.Adam(rnn_list[i].parameters(), lr=network_config["training"]["lr"]))
        scheduler_list.append(MultiStepLR(optimizer_list[i], milestones=network_config["training"]["scheduler"]["milestones"], 
                                          gamma=network_config["training"]["scheduler"]["gamma"]))

    # loss function
    loss_func = nn.MSELoss()
    loss_func_l1 = nn.L1Loss()
    MaxlikihoodLoss = nn.GaussianNLLLoss()

    def loss_distribution_diag(pred, pred_logstd, targ):
        MIN_LOG_STD = np.log(1e-6)

        pred_logstd = torch.maximum(pred_logstd, MIN_LOG_STD * torch.ones_like(pred_logstd))
        loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd
        return loss
    
    def loss_mse_seq(pred, targ):
        # output dim  = (batch, seq, 1)
        loss = torch.mean( (pred - targ).pow(2) , axis = (0,2) )
        return loss

    loss_list1 = []
    test_loss_list1 = []
    test_gnll_loss_list = []

    iter = network_config["training"]["start_iter"]
    test_loss = None
    mse_max_iter = network_config["training"]["gnll_start_iter"] # max iterations for MSE loss
    tapping_point = 40

    fig, ax = plt.subplots(3, 1)
    seq_fig = ax[0]
    uncertainity_fig = ax[1]
    loss_fig = ax[2]
    
    root_dir = network_config["utils"]["root_dir"]
    if not os.path.exists(root_dir):
        print("root dir %s not found, exiting", root_dir)
        exit()
    else:            
        save_dir = os.path.join(root_dir, "model", network_config["utils"]["savename"])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
    # log the loss print statements in a file
    if network_config["utils"]["save_model"]:
        log_file = open(os.path.join(network_config["utils"]["root_dir"], "model", network_config["utils"]["savename"], "log.txt"), "w")
        log_file.flush()

    while(True):

        for i, data in enumerate(train_slice):
                        
            input_data_mask = network_config["network"]["input_data_mask"]
            
            input_data = input_data_gen(data, network_config, train=True)
            
            # output_data = data["supervision_normed"][:, -1:, :3].cuda() - data["supervision_normed"][:, tapping_point:tapping_point+1, :3].cuda()
            output_data = data["supervision_normed"][:, -1:, :3].cuda()
            
            vel_init = data["supervision_normed"][:, :1, :3].cuda()

            # biases_init = data["biases_normed"][:, -1, :6].cuda()
            # vel_biases_int = torch.cat([vel_init,biases_init ], dim=-1)                
            
            # information_mask = torch.randint(0, 2, (input_data.shape[0],)).unsqueeze(-1).unsqueeze(-1).cuda()            
            # # if iter>=25000 and iter<35000:
            # if iter<250:
            #     input_data[:,:,6:] = information_mask*input_data[:,:,6:]

            # optimizer.zero_grad()
            for i in range(ensemble_size):
                optimizer = optimizer_list[i]
                scheduler = scheduler_list[i]
                rnn = rnn_list[i]
            
                rnn.train()
                output_data_pred_full, uncertainity_full= rnn(input_data)
                output_data_pred = output_data_pred_full[:, -1:, :]
                uncertainity = uncertainity_full[:, -1:, :]
                # output_data_pred, uncertainity, body_f_dp= rnn(data["motor_command_normed"][:,:,:8].cuda(), data["alphasense_imu_frag_normed"][:, :,:6].cuda(), vel_init= vel_init)
                
                # get loss
                loss_train = loss_func(output_data_pred, output_data )
                
                # formulate gaussian likelihood loss
                loss_gnll = torch.mean( loss_distribution_diag(output_data_pred, uncertainity, output_data) )
                
                # total loss
                if iter<mse_max_iter:
                    loss = loss_func(output_data_pred, output_data)
                else:
                    loss = loss_gnll 
                
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i == 0:
                    loss_list1.append([iter, loss_train.item()])

            iter+=1
            if iter> network_config["training"]["max_iter"]:
                print("Training done")
                total_time = time.time() - training_start_time
                print("Total time taken: ", total_time/60, " minutes")
                break

            # test
            if iter%20 ==0: # test only after every 10 iterations
                for i, data in enumerate( test_slice ):
                    
                    input_data = input_data_gen(data, network_config)
                    
                    output_data = data["supervision_normed"][:, -1:, :3].cuda()
                    vel_init = data["supervision_normed"][:, :1, :3].cuda()
                    
                    # biases_init = data["biases_normed"][:, :1, :6].cuda()
                    # vel_biases_int = torch.cat([vel_init,biases_init ], dim=-1)  

                    # cuda and double
                    rnn.eval()
                    with torch.no_grad():                        
                        output_data_pred_full, uncertainity_full= rnn(input_data)
                        output_data_pred = output_data_pred_full[:, -1:, :]
                        uncertainity = uncertainity_full[:, -1:, :]
                        # output_data_pred, uncertainity, body_f_dp= rnn(data["motor_command_normed"][:,:,:8].cuda(), data["alphasense_imu_frag_normed"][:, :,:6].cuda(), vel_init= vel_init)

                        # get loss
                        # test_loss = loss_func(output_data_pred, output_data)
                        test_loss = loss_func(output_data_pred , output_data )
                        test_loss_list1.append([iter, test_loss.item()])
                        
                        seqlen_equalize_factor = int(output_data_pred_full.shape[1]/data["supervision_normed"].shape[1])
                        mse_seq_loss = loss_mse_seq(output_data_pred_full[:, ::seqlen_equalize_factor], data["supervision_normed"][:, :, :3].cuda())
                        uncertainity_seq = uncertainity_full[:, :, :]
                        
                        mse_seq_loss_np = mse_seq_loss.cpu().numpy()
                        
                        #plot mse_seq_loss
                        
                        # plot the sequence loss

                        seq_fig.clear()                        
                        seq_fig.plot(mse_seq_loss_np, label="MSE loss", color="red", alpha = 0.8)

                        seq_fig.grid()
                        seq_fig.legend()
                        # seq_fig.set_title("Sequence MSE Loss")
                        seq_fig.set_ylabel("MSE Loss")
                        seq_fig.set_xlabel("Sequence")
                        # seq_fig.set_ylim(0, 2.5)
                        
                        # plot the uncertainity of the sequence
                        uncertainity_seq_np = uncertainity_seq.cpu().numpy()
                        
                        uncertainity_fig.clear()
                        
                        mean_uncertainity = np.mean(uncertainity_seq_np, axis = 0)
                        
                        uncertainity_fig.plot(mean_uncertainity[:, 0], label="Mean Uncertainity x", color="red", alpha = 0.8, linestyle = "--")
                        uncertainity_fig.plot(mean_uncertainity[:, 1], label="Mean Uncertainity y", color="green", alpha = 0.8, linestyle = "--")
                        uncertainity_fig.plot(mean_uncertainity[:, 2], label="Mean Uncertainity z", color="blue", alpha = 0.8, linestyle = "--")
                        
                        uncertainity_fig.grid()
                        uncertainity_fig.legend()
                        # uncertainity_fig.set_title("Mean Uncertainity")
                        uncertainity_fig.set_ylabel("Uncertainity")
                        uncertainity_fig.set_xlabel("Sequence Length")
                        # uncertainity_fig.set_ylim(0, 0.5)
                        
                        test_loss_gnll = torch.mean( loss_distribution_diag(output_data_pred, uncertainity, output_data) )
                        test_gnll_loss_list.append([iter, test_loss_gnll.item()])
                    # break randomly
                    if np.random.rand()>0.2:
                        break
        
            if test_loss is not None:
                scheduler.step(test_loss)
                
                # plot loss
                # plt.clf()
                
                loss_list1_np = np.array(loss_list1)
                test_loss_list1_np = np.array(test_loss_list1)
                test_gnll_loss_list_np = np.array(test_gnll_loss_list)

                if iter%100 == 0:
                    print("Iter: ", iter, " loss: ", loss_train.item(), " test_loss: ", test_loss.item())
                    print("Iter: ", iter, " smooth loss: ", np.mean(loss_list1_np[-50:, 1]), " smooth test_loss: ", np.mean(test_loss_list1_np[-50:, 1]))
                    log_file.write("Iter: " + str(iter) + " loss: " + str(loss_train.item()) + " test_loss: " + str(test_loss.item()) + "\n")
                    log_file.write("Iter: " + str(iter) + " smooth loss: " + str(np.mean(loss_list1_np[-50:, 1])) + " smooth test_loss: " + str(np.mean(test_loss_list1_np[-50:, 1])) + "\n")
                    log_file.flush()
                
                loss_fig.clear()
                loss_fig.plot(loss_list1_np[:, 0], loss_list1_np[:, 1], label="Train loss", color="red", alpha = 0.8)
                loss_fig.plot(test_loss_list1_np[:, 0], test_loss_list1_np[:, 1], label="Test loss", color="blue", alpha = 0.8)
                loss_fig.plot(test_gnll_loss_list_np[:, 0], test_gnll_loss_list_np[:, 1], label="Test GNLL loss", color="blue", alpha = 0.8, linestyle = "--")
                loss_fig.grid()
                loss_fig.legend()
                loss_fig.set_xlabel("Iteration")
                loss_fig.set_ylabel("Loss")
                # loss_fig.set_ylim(0, 4)
                
                if iter%100 == 0:
                    plt.pause(0.01)
                    
                if iter%200 == 0:
                    # save the model
                    if network_config["utils"]["save_model"]:
                        for i in range(ensemble_size):
                            torch.save(rnn_list[i].state_dict(), os.path.join(save_dir, "net_"+str(i)+".pt"))
                        plt.savefig(os.path.join(save_dir, "plot.png"))
                        yaml.dump(network_config, open(os.path.join(save_dir, "config.yaml"), "w"), sort_keys=False)
                        
                        print("Model saved at iteration: ", iter)
                    
        if iter> network_config["training"]["max_iter"]:
            break
    
    if network_config["utils"]["save_model"]:    
        # make directory if not exist for saving all the models in the ensemble
        root_dir = network_config["utils"]["root_dir"]
        if not os.path.exists(root_dir):
            print("root dir %s not found, exiting", root_dir)
            exit()
        else:            
            save_dir = os.path.join(root_dir, "model", network_config["utils"]["savename"])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            for i in range(ensemble_size):
                torch.save(rnn_list[i].state_dict(), os.path.join(save_dir, "net_"+str(i)+".pt"))

            plt.savefig(os.path.join(save_dir, "plot.png"))

            repo = git.Repo(search_parent_directories=True)        
            network_config["git"]["commit"] = repo.head.object.hexsha
            network_config["git"]["branch"] = repo.active_branch.name
            network_config["git"]["repo"] = repo.remotes.origin.url
            network_config["git"]["commit_message"] = repo.head.object.message
            
            # save the config file
            yaml.dump(network_config, open(os.path.join(save_dir, "config.yaml"), "w"), sort_keys=False)
