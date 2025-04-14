#!/usr/bin/env python
"""
Copyright (c) 2025, Autonomous Robots Lab, Norwegian University of Science and Technology
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks.networks import TemporalConvNet, TCN, RNN
from dataset.Dataset import dbioDataset

import rospy
from sensor_msgs.msg import Imu
from message_filters import TimeSynchronizer, Subscriber
from mavros_msgs.msg import RCOut
from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import TwistStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

import argparse

import message_filters

import time
import yaml
from dataset.Dataset import dbioDataset
from os import path


class LearnedInertialOdom:
    def __init__(self, imu_topic=None, fcu_imu_topic=None, act_topic=None, battery_topic=None, device="cpu", network_config=None, use_fcimu_time=False, verbose=False):
        
        self.device = torch.device(device)
        self.use_fcimu_time = use_fcimu_time
        self.verbose = verbose
        
        self.gravity_correction = network_config["dataset"]["gravity_correction"]
        self.input_mask = network_config["network"]["input_data_mask"]

        # Buffers
        seq_duration = network_config["dataset"]["seq_duration"]
        max_seq_duration = 2
        
        imu_fps = network_config["dataset"]["sensor_fps"]["alphasense_imu"]
        fcu_imu_fps = network_config["dataset"]["sensor_fps"]["fcu_imu"]
        act_fps = network_config["dataset"]["sensor_fps"]["motor_command"]
        velocity_fps = network_config["dataset"]["sensor_fps"]["rovio_odom_twist"]
        battery_fps = network_config["dataset"]["sensor_fps"]["battery"]
        
        fragment_imu_length_full = imu_fps*max_seq_duration 
        fragment_fcu_imu_length_full = fcu_imu_fps*max_seq_duration 
        fragment_act_length_full = act_fps*max_seq_duration  
        fragment_vel_length_full = velocity_fps*max_seq_duration
        fragment_battery_length_full = battery_fps*max_seq_duration
                
        self.imu_buffer_full = torch.zeros((1, fragment_imu_length_full, 7), device = self.device, dtype=torch.float64)
        self.fcu_imu_buffer_full  = torch.zeros((1, fragment_fcu_imu_length_full, 7), device = self.device)
        self.act_buffer_full =  torch.zeros((1, fragment_act_length_full, 9), device = self.device)
        self.velocity_buffer_full = torch.zeros((1, fragment_vel_length_full, 7), device = self.device)
        self.battery_buffer_full = torch.zeros((1, fragment_battery_length_full, 3), device = self.device)
        self.elapsed_time_buffer = []
        
        self.ensemble_size = network_config["network"]["ensemble_size"]
        self.uncertainty_type = network_config["testing"]["uncertainty_type"]
        self.rnn_list = []

        self.alphasense_imu_mean = torch.tensor(network_config["dataset"]["sensor_stats"]["alphasense_imu"]["mean"], device = self.device).unsqueeze(0).unsqueeze(0)
        self.alphasense_imu_std = torch.tensor(network_config["dataset"]["sensor_stats"]["alphasense_imu"]["std"], device = self.device).unsqueeze(0).unsqueeze(0)
        
        self.fcu_imu_mean = torch.tensor(network_config["dataset"]["sensor_stats"]["fcu_imu"]["mean"], device = self.device).unsqueeze(0).unsqueeze(0)
        self.fcu_imu_std = torch.tensor(network_config["dataset"]["sensor_stats"]["fcu_imu"]["std"], device = self.device).unsqueeze(0).unsqueeze(0)
        
        self.actuation_mean = torch.tensor(network_config["dataset"]["sensor_stats"]["motor_command"]["mean"], device = self.device).unsqueeze(0).unsqueeze(0)
        self.actuation_std = torch.tensor(network_config["dataset"]["sensor_stats"]["motor_command"]["std"], device = self.device).unsqueeze(0).unsqueeze(0)
        
        self.vel_mean = torch.tensor(network_config["dataset"]["sensor_stats"]["rovio_odom_twist"]["mean"], device = self.device).unsqueeze(0).unsqueeze(0)
        self.vel_std = torch.tensor(network_config["dataset"]["sensor_stats"]["rovio_odom_twist"]["std"], device = self.device).unsqueeze(0).unsqueeze(0)
        
        self.battery_mean = torch.tensor(network_config["dataset"]["sensor_stats"]["battery"]["mean"], device = self.device).unsqueeze(0).unsqueeze(0)
        self.battery_std = torch.tensor(network_config["dataset"]["sensor_stats"]["battery"]["std"], device = self.device).unsqueeze(0).unsqueeze(0)
        
        self.cb_counter = 1
        self.cb_counter_synced = 1
        self.max_buffer_len = 160 # 20Hz data
        self.battery_callback_counter = 1
        
        # RNN based network
        self.prev_hidden_state = None
        self.prev_hidden_state_list = [None]*self.ensemble_size
        self.vel_init = None
        self.vel_init_mean = self.vel_mean
        self.vel_init_std = self.vel_std

        self.imu_biases = np.zeros(6)
        self.odom_time_stamp = 0
        
        for i in range(self.ensemble_size):
            self.rnn_list.append(RNN(network_config["network"]["input_size"], 
                                network_config["network"]["output_size"], 
                                network_config["network"]["layers"], 
                                3, 
                                network_config["network"]["dropout"], real_time=True,
                                skip_connection=network_config["network"]["skip_connection"]).to(self.device))
            
            self.rnn_list[i]
            self.rnn_list[i].load_state_dict(torch.load(ENSEMBLE_DIR_PATH+"/net_"+str(i)+".pt"))
            self.rnn_list[i].eval()
            print("Loaded model", i)
            
        # Subscribers and Publishers                        
        self.odom_sub = rospy.Subscriber("/rovio/odometry", Odometry, self.callback_odom)
        self.baseline_twist_pub = rospy.Publisher("/baseline_twist", TwistStamped, queue_size=10)
        
        self.baseline_twist_cov_pub = rospy.Publisher("/baseline_twist_cov", TwistWithCovarianceStamped, queue_size=10)

        self.vel_pub = rospy.Publisher("/abss/twist", TwistStamped, queue_size=10)
        self.vel_cov_pub = rospy.Publisher("/abss_cov/twist", TwistWithCovarianceStamped, queue_size=10)
        self.vel_cov_epistemic_pub = rospy.Publisher("/deepvl_vel_cov/twist", TwistWithCovarianceStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/velocity_covariance", Marker, queue_size = 2)
               
        self.imu_sub_sync = rospy.Subscriber(imu_topic, Imu, self.callback_imu)        
        self.act_sub_sync = rospy.Subscriber(act_topic, RCOut, self.callback_motor_cmd_sync)
        self.fcu_imu_sub_sync = rospy.Subscriber(fcu_imu_topic, Imu, self.callback_fcu_imu)
        self.battery_sub = rospy.Subscriber(battery_topic, BatteryState, self.callback_battery)
        
        self.act_data = None
        self.imu_data = None
        self.fcu_imu_data = None
        self.fcimu_time_stamp = 0.0
        
        self.current_voltage = 14.8
    
    def callback_battery(self, msg):
        self.battery_buffer_full = torch.roll(self.battery_buffer_full, -1, 1)
        self.battery_buffer_full[0, -1, :] = torch.tensor([1.0, msg.voltage, msg.header.stamp.to_sec()], device = self.device)
        self.battery_buffer_full[0, -1, -1] = msg.header.stamp.to_sec()
        self.battery_callback_counter+=1
        self.current_voltage = msg.voltage
        
    def callback_imu(self, msg):
        self.imu_buffer_full = torch.roll(self.imu_buffer_full, -1, 1)
        self.imu_buffer_full[0, -1, :] = torch.tensor([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.header.stamp.to_sec()], device = self.device)
        self.imu_buffer_full[0, -1, -1] = msg.header.stamp.to_sec()
        
    def callback_fcu_imu(self, msg):
        self.fcimu_time_stamp = msg.header.stamp.to_sec()
        self.fcu_imu_buffer_full = torch.roll(self.fcu_imu_buffer_full, -1, 1)
        self.fcu_imu_buffer_full[0, -1, :] = torch.tensor([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.header.stamp.to_sec()], device = self.device)
        self.fcu_imu_buffer_full[0, -1, -1] = msg.header.stamp.to_sec()
        
    def callback_odom(self, msg):
        # this callback is used only for visualization of the baseline twist to enable comparision of the deepvl twist visually in rviz
        twist_msg = TwistStamped()
        twist_msg.header = msg.header
        twist_msg.header.frame_id = "imu"
        twist_msg.twist.linear = msg.twist.twist.linear
        self.baseline_twist_pub.publish(twist_msg)
        
        # identity covariance matrix
        cov_matrix = np.eye(6)
        cov_matrix36 = cov_matrix.reshape(36)
        
        twist_cov_msg = TwistWithCovarianceStamped()
        twist_cov_msg.header = msg.header
        twist_cov_msg.header.frame_id = "imu"
        twist_cov_msg.twist.twist.linear = msg.twist.twist.linear
        
        twist_cov_msg.twist.covariance = cov_matrix36
        
        self.baseline_twist_cov_pub.publish(twist_cov_msg)

    def callback_motor_cmd_sync(self, act_msg):
        full_start_time = time.time()
        start_time = time.time()
        
        if self.use_fcimu_time and self.fcimu_time_stamp > 0.0:
            # ros warning
            if verbose: rospy.logwarn("Using FCU IMU time stamp, not recommended")
            act_msg.header.stamp = rospy.Time(self.fcimu_time_stamp)
        
        self.act_data = torch.tensor([act_msg.channels[0], 
                                act_msg.channels[1],
                                act_msg.channels[2],
                                act_msg.channels[3],
                                act_msg.channels[4],
                                act_msg.channels[5],
                                act_msg.channels[6],
                                act_msg.channels[7]], device = self.device)
        
        fcu_imu_data_sync_idx = torch.argmin(torch.abs(self.fcu_imu_buffer_full[0, :, -1] - act_msg.header.stamp.to_sec()))
        self.fcu_imu_data = self.fcu_imu_buffer_full[:, fcu_imu_data_sync_idx, :6]
        
        imu_data_sync_idx = torch.argmin(torch.abs(self.imu_buffer_full[0, :, -1] - act_msg.header.stamp.to_sec()))
        time_diff = self.imu_buffer_full[0, imu_data_sync_idx, -1] - act_msg.header.stamp.to_sec()
        if verbose: print("time diff", time_diff.item())
        if verbose: print("imu_data_sync_idx", imu_data_sync_idx.item())

        if abs(time_diff) < 1.0:
            self.imu_data = self.imu_buffer_full[:, imu_data_sync_idx-10:imu_data_sync_idx, :6]
            self.imu_data = torch.mean(self.imu_data, dim=1)
            # convert to float32
            self.imu_data = self.imu_data.float()
            
            # battery_data_sync_idx = torch.argmin(torch.abs(self.battery_buffer_full[0, :, -1] - act_msg.header.stamp.to_sec()))
            self.battery_data = self.battery_buffer_full[:, -1, :2]
            
            self.act_data = self.act_data.view(1,1,-1)
            self.fcu_imu_data = self.fcu_imu_data.view(1,1,-1)
            self.imu_data = self.imu_data.view(1,1,-1)
            self.battery_data = self.battery_data.view(1,1,-1)
            
            # normalize
            self.imu_data_normed = (self.imu_data - self.alphasense_imu_mean[0])/self.alphasense_imu_std[0]
            self.act_data_normed = (self.act_data - self.actuation_mean[0])/ self.actuation_std[0]
            self.fcu_imu_data_normed = (self.fcu_imu_data - self.fcu_imu_mean[0])/self.fcu_imu_std[0]
            self.battery_data_normed = (self.battery_data - self.battery_mean[0])/self.battery_std[0]
            
            self.battery_data_normed[:,:,0] = 1.0
            
            # battery data negated with input mask if the battery data is not available
            if self.battery_callback_counter < 2:
                self.input_mask[2] = 0
            
            self.input_data = torch.cat((self.input_mask[0]*self.imu_data_normed,
                                        self.input_mask[1]*self.act_data_normed,
                                        self.input_mask[2]*self.battery_data_normed), dim = -1)
            
            # convert input data to float32
            self.input_data = self.input_data.float()

            data_conversion_time = time.time() - start_time
            # if verbose: print("input data conversion time", data_conversion_time*1000)
            start_time = time.time()
                                
            output_data_pred_list = []
            uncertainity_list = []
            h_list = []
            
            if self.uncertainty_type != "epistemic":
                rospy.logwarn("Uncertainty type not from ensemble")
                            
            with torch.no_grad():
                for i in range(self.ensemble_size):
                    output_data_pred, uncertainity, h_= self.rnn_list[i](self.input_data, self.prev_hidden_state_list[i])
                    self.rnn_list[i].rnn_vel_initialized = True
                    output_data_pred_list.append(output_data_pred)
                    uncertainity_list.append(uncertainity)
                    h_list.append(h_)
            
            inference_time = time.time() - start_time
            # if verbose: print("inference time", inference_time*1000)
            start_time = time.time()

            output_data_pred_stack = torch.stack(output_data_pred_list)
            output_data_pred_ = torch.mean(output_data_pred_stack, dim=0)  # mean of the ensemble output (mu^{2}_{*}(x))
            uncertainity_stack = torch.exp(2*torch.stack(uncertainity_list)) # variance of the ensemble output (sigma^{2}_{*}(x))
            sigma_ensemble = torch.mean(output_data_pred_stack**2 + uncertainity_stack, dim=0) - output_data_pred_**2
            cov_epistemic = sigma_ensemble.detach().cpu().numpy()
            
            output_data_pred_ = (output_data_pred_*self.vel_std[:, :, :3]) + self.vel_mean[:, :, :3]
            vel_pred = output_data_pred_.detach().cpu().numpy()
            cov = np.exp(2*uncertainity.detach().cpu().numpy())

            output_data_no_ensemble = (output_data_pred*self.vel_std[:, :, :3]) + self.vel_mean[:, :, :3]
            vel_no_ensemble_pred = output_data_no_ensemble.detach().cpu().numpy()
            
            vel_msg = TwistStamped()
            vel_no_ensemble_msg = TwistWithCovarianceStamped()
            vel_cov_msg = TwistWithCovarianceStamped()
            vel_cov_epistemic_msg = TwistWithCovarianceStamped()
            
            voltage_scaling = 1 + (self.current_voltage-14.8)/14.8
            vel_msg.header.stamp = act_msg.header.stamp
            vel_msg.header.frame_id = "imu"
            vel_msg.twist.linear.x = vel_pred[0,0,0]
            vel_msg.twist.linear.y = vel_pred[0,0,1]
            vel_msg.twist.linear.z = vel_pred[0,0,2]

            vel_no_ensemble_msg.header.stamp = act_msg.header.stamp
            vel_no_ensemble_msg.header.frame_id = "imu"
            vel_no_ensemble_msg.twist.twist.linear.x = vel_no_ensemble_pred[0,0,0]
            vel_no_ensemble_msg.twist.twist.linear.y = vel_no_ensemble_pred[0,0,1]
            vel_no_ensemble_msg.twist.twist.linear.z = vel_no_ensemble_pred[0,0,2]
            
            cov_scaler_x = 1.0
            cov_scaler_y = 1.0
            cov_scaler_z = 1.0
        
            cov_matrix = np.diag([cov[0,0,0]*cov_scaler_x, 
                                cov[0,0,1]*cov_scaler_y, 
                                cov[0,0,2]*cov_scaler_z, 
                                0.0, 0.0, 0.0])

            cov_matrix36 = cov_matrix.reshape(36)
            
            vel_cov_msg.header.stamp = act_msg.header.stamp
            vel_cov_msg.header.frame_id = "imu"     
            vel_cov_msg.twist.twist = vel_msg.twist
            
            vel_cov_msg.twist.covariance = cov_matrix36
            
            vel_no_ensemble_msg.twist.covariance = cov_matrix36
            
            vel_cov_epistemic_msg.header.stamp = act_msg.header.stamp
            vel_cov_epistemic_msg.header.frame_id = "imu"
            vel_cov_epistemic_msg.twist.twist = vel_msg.twist
            
            cov_epistemic_scale = 1.0
            
            cov_matrix_epistemic = np.diag([cov_epistemic_scale*cov_epistemic[0,0,0],
                                            cov_epistemic_scale*cov_epistemic[0,0,1],
                                            cov_epistemic_scale*cov_epistemic[0,0,2],
                                            0.0, 0.0, 0.0])
            
            cov_matrix36_epistemic = cov_matrix_epistemic.reshape(36)
            # cov_matrix36_epistemic = cov_matrix_identity.reshape(36)
            vel_cov_epistemic_msg.twist.covariance = cov_matrix36_epistemic
                
            output_data_time = time.time() - start_time 
            # if self.verbose: print("output_data_time", output_data_time*1000)
            full_time = time.time() - full_start_time
            if len(self.elapsed_time_buffer) > 200:
                self.elapsed_time_buffer.pop(0)
            self.elapsed_time_buffer.append(full_time)
            mean_elapsed_time = np.mean(self.elapsed_time_buffer)
            
            if self.verbose: print("mean_elapsed_time", mean_elapsed_time*1000)

            if self.rnn_list[0].rnn_vel_initialized:
                
                for i in range(self.ensemble_size):
                    self.prev_hidden_state_list[i] = h_list[i]
                
                # if self.cb_counter_synced%10 == 0:
                self.vel_pub.publish(vel_msg)
                self.vel_cov_pub.publish(vel_cov_msg)
                if self.uncertainty_type == "epistemic":
                    self.vel_cov_epistemic_pub.publish(vel_cov_epistemic_msg)
                elif self.uncertainty_type == "network":
                    self.vel_cov_epistemic_pub.publish(vel_no_ensemble_msg)
                self.publish_cov_marker(cov[0,0], vel_cov_msg.header.stamp)
                self.publish_cov_marker(cov_epistemic[0,0], vel_cov_epistemic_msg.header.stamp, id = 1, color = [0.75, 0.75, 1.0, 0.6])
                if self.verbose: print("Velocity prediction", vel_pred[0,0])
            
                self.cb_counter_synced+=1        
        
    def publish_cov_marker(self, cov, timestamp, id = 0, color = [1.0, 0.75, 0.75, 0.6]):
        marker = Marker()

        marker.header.frame_id = "imu"
        # marker.child_frame_id = "/imu"
        marker.header.stamp = timestamp

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 2
        marker.id = id

        # Set the scale of the marker
        scale = 10.0
        marker.scale.x = cov[0]*scale
        marker.scale.y = cov[1]*scale
        marker.scale.z = cov[2]*scale

        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        # Set the pose of the marker
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        self.marker_pub.publish(marker)
        
    def spin(self):
        rospy.spin()

if __name__=="__main__":
    rospy.init_node("deepvl_ros_node")
    
    system_1_root = "/home/sakura/rosworkspaces/catkin_ws/src/deepvl"
    system_2_root = "/home/ariel/catkin_ws/src/inertial_odom"
    system_subdir = "scripts/model"
    
    model_name = "training_full_raw_data_baseline_fcimu"
    model_name = rospy.get_param("~model_name", model_name)
    imu_topic = rospy.get_param("~imu_topic", "/alphasense_driver_ros/imu_drop")
    fcu_imu_topic = rospy.get_param("~fcu_imu_topic", "/mavros/imu/data_drop")
    act_topic = rospy.get_param("~act_topic", "/mavros/rc/out_drop")
    battery_topic = rospy.get_param("~battery_topic", "/mavros/battery")
    use_fcimu_time = rospy.get_param("~use_fcimu_time", False)
    verbose = rospy.get_param("~verbose", False)
    
    device = rospy.get_param("~device", "cpu")
    
    if path.exists(system_1_root):
        ENSEMBLE_DIR_PATH = path.join(system_1_root, system_subdir, model_name)
    elif path.exists(system_2_root):
        ENSEMBLE_DIR_PATH = path.join(system_2_root, system_subdir, model_name)
    
    network_config = yaml.load(open(path.join(ENSEMBLE_DIR_PATH, "config.yaml")), Loader=yaml.FullLoader)
    
    imu_sync = LearnedInertialOdom(imu_topic=imu_topic,
                                   fcu_imu_topic=fcu_imu_topic, 
                                   act_topic=act_topic,
                                   battery_topic=battery_topic,
                                   device=device,
                                   network_config=network_config, 
                                   use_fcimu_time=use_fcimu_time,
                                   verbose=verbose)
    imu_sync.spin()

