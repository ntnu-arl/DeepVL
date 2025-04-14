import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rospy
import time

from matplotlib.patches import Ellipse

from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

from networks.networks import TemporalConvNet

from torch.utils.data import Dataset, DataLoader
from dataset.Dataset import dbioDataset

test_dataset_dir = "/home/sakura/ssd/Datasets/MClabDataset/DBIO/test"
test_dataset = dbioDataset(test_dataset_dir,seq_duration=8.0, train=False, traj_name = "traj15")

# # TCN 

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.linear_vel = nn.Linear(num_channels[-1], output_size)
        self.linear_std = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out.permute(0, 2, 1)
        
        vel = self.linear_vel(out[:, -1:, :])
        std = self.linear_std(out[:, -1:, :])
        return vel, std

# parameters to double
# rnn.double()
rnn = TCN(14, 3, [14, 14, 16, 16, 32, 32, 32], 3, 0.2).cuda()
rnn.double()


# count number of params in the network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total Param count: ", count_parameters(rnn))

def get_receptive(kernel_size, levels, dilation_exponential_base): 
    return sum([dilation_exponential_base**(l-1)*(kernel_size-1) for l in range(levels, 0, -1)]) + 1

print("receptive field: ", get_receptive(2, 8, 2))

# load the model
PATH = "/home/sakura/catkin_ws/src/inertial_odom/scripts/model/TCN_16traj_l1_lrspet_dataaug.pt"
rnn.load_state_dict(torch.load(PATH))
rnn.eval()

alphasense_imu = test_dataset.test_data["alphasense_imu"]
accuator_command_data = test_dataset.test_data["accuator_command"]
velocity_est = test_dataset.test_data["supervision"]

start_time = velocity_est[0, -1:]
end_time = velocity_est[-1, -1:]

offset_time = 20
t = start_time + offset_time 

fragment_length = 8.0 # seconds


dt = 0.05
passed_count = 0
failed_count = 0

fig, ax = plt.subplots(figsize = (10, 10))

count = 0
vel_pred_list = []
vel_rovio_list = []
early_terminate = False
while True:
    alphasense_imu_fragment_ = test_dataset.get_fragment(fragment_length*test_dataset.alphasense_imu_fps, alphasense_imu, t)
    # print("alphasense_imu_fragment", alphasense_imu_fragment_.shape, alphasense_imu.shape)
    
    if alphasense_imu_fragment_.shape[0]!=1600:
        failed_count+=1
        t = t+dt
        
        continue
    alphasense_imu_fragment = np.mean(alphasense_imu_fragment_.reshape(-1, 10, 7), axis=1)

    
    alphasense_imu_fragment = test_dataset.normalize_data(alphasense_imu_fragment, test_dataset.alphasense_imu_mean, test_dataset.alphasense_imu_std)

    accuator_command_fragment_raw = test_dataset.get_fragment(fragment_length*test_dataset.accuator_command_fps, accuator_command_data, t)
    accuator_command_fragment = test_dataset.normalize_data(accuator_command_fragment_raw, test_dataset.accuator_command_mean, test_dataset.accuator_command_std)


    velocity_est_fragment_raw = test_dataset.get_fragment(fragment_length*test_dataset.supervision_fps, velocity_est, t)
    velocity_est_fragment = test_dataset.normalize_data(velocity_est_fragment_raw, test_dataset.supervision_mean, test_dataset.supervision_std)

    alphasense_imu_fragment = torch.tensor(alphasense_imu_fragment).unsqueeze(0).double().cuda()
    accuator_command_fragment = torch.tensor(accuator_command_fragment).unsqueeze(0).double().cuda()
    velocity_est_fragment = torch.tensor(velocity_est_fragment).unsqueeze(0).double().cuda()

    if alphasense_imu_fragment.shape[1] == accuator_command_fragment.shape[1] == velocity_est_fragment.shape[1]:
        # print("alphasense_imu_fragment", alphasense_imu_fragment.shape)
        # print("accuator_command_fragment", accuator_command_fragment.shape)
        # # print("velocity_est_fragment", velocity_est_fragment.shape)
        passed_count+=1
    else: 
        failed_count+=1
        t = t+dt
        
        continue
        
    t = t+dt

    # print("passed_count:   ",passed_count,  "   failed_count: ", failed_count)
    
    if t> end_time:
        break
        
    imu_mask = 1
    accu_mask = 1

    input_data = torch.cat((imu_mask*alphasense_imu_fragment[:, :,:6], accu_mask*accuator_command_fragment[:,:,:8]), dim=2)
    output_data = velocity_est_fragment[:, -1:, :3]
    vel_rovio = velocity_est_fragment_raw[-1:, :3]

    # # cuda and double
    with torch.no_grad():
        output_data = output_data.double().cuda()

        start_time = time.time()   

        input_data = input_data.double().cuda()
        
        output_data_pred, uncertainity= rnn(input_data)
        
        output_data_pred = output_data_pred.detach().cpu().numpy()
        uncertainity = uncertainity.detach().cpu().numpy()
        covariance = np.exp(2*uncertainity)
        elapsed_time = time.time() - start_time

        print("elapsed_time:  ", elapsed_time*1000)

        output_data = output_data.detach().cpu().numpy()

        vel_pred = (output_data_pred[0]*test_dataset.supervision_std[:3]) + test_dataset.supervision_mean[:3]
        # std_ = np.sqrt(covariance)
        std_ = np.sqrt(covariance)
        vel_cov_t = np.concatenate([vel_pred, std_[0], accuator_command_fragment[:,-1:,8].cpu().numpy()], axis=-1)

        vel_pred_list.append(vel_cov_t)
        vel_rovio_list.append(vel_rovio)
        # Creating arrow
        x_pos = 0
        y_pos = 0
        # Creating plot
        ax.clear()
        
        ax.quiver(x_pos, y_pos, vel_rovio[0,0], vel_rovio[0,1], color="k", alpha=0.5, angles='xy', scale_units='xy', scale=1, label="Velocity, Prediction (TCN)") # prediction

        ax.quiver(x_pos, y_pos, vel_pred[0,0], vel_pred[0,1], color="r", alpha=0.8, angles='xy', scale_units='xy', scale=1, label="Velocity, Rovio est (GT)") # prediction
        
        ells = Ellipse(xy=[0,0], width=std_[0,0, 0], height=std_[0,0, 1], angle=0, label = "Uncertainity")
 
        ax.set_title('Learning Based Proprioceptive Odometry')
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_xlabel("X $(m/s)$ in IMU frame")
        ax.set_ylabel("Y $(m/s)$ in IMU frame")
        
        ax.add_artist(ells)
        ells.set_clip_box(ax.bbox)
        ells.set_alpha(0.3)
        ells.set_facecolor('g')
        plt.legend()
        plt.grid()

        # # Show plot
        if count ==0 and not early_terminate:
            plt.pause(5)
        else:
            plt.pause(0.001)
            
        count +=1
        
        if count > 3000 and early_terminate:
            break
plt.show()

vel_pred_list_ = np.array(vel_pred_list)
vel_rovio_list_ = np.array(vel_rovio_list)
vel_pred_list_[:, :, -1] = vel_pred_list_[:, :, -1] - vel_pred_list_[:1, :, -1]
print("vel_pred_list.shape", vel_pred_list_[:, :, -1])    
print("vel_rovio_list_ shape", vel_rovio_list_.shape)

fig, ax = plt.subplots(figsize = (10, 10))

ax.plot(vel_pred_list_[:, 0, -1], vel_pred_list_[:, 0, 0] , label="X (Prediction)", color="r")
ax.plot(vel_pred_list_[:, 0, -1], vel_rovio_list_[:, 0, 0] , label="X (ROVIO)", color="r", linestyle="--")
upper_bound = vel_pred_list_[:, 0, 0] - vel_pred_list_[:, 0, 3]
lower_bound = vel_pred_list_[:, 0, 0] + vel_pred_list_[:, 0, 3]

# plt.fill_between(vel_pred_list_[:, 0, -1], upper_bound, lower_bound, color='r', alpha=.3)

ax.plot(vel_pred_list_[:, 0, -1], vel_pred_list_[:, 0, 1] , label="Y (Prediction)", color="g")
ax.plot(vel_pred_list_[:, 0, -1], vel_rovio_list_[:, 0, 1] , label="Y (ROVIO)", color="g", linestyle="--")

ax.plot(vel_pred_list_[:, 0, -1], vel_pred_list_[:, 0, 2] , label="Z (Prediction)", color="b")
ax.plot(vel_pred_list_[:, 0, -1], vel_rovio_list_[:, 0, 2] , label="Z (ROVIO)", color="b", linestyle="--")


upper_bound = vel_pred_list_[:, 0, 1] - vel_pred_list_[:, 0, 4]
lower_bound = vel_pred_list_[:, 0, 1] + vel_pred_list_[:, 0, 4]

# plt.fill_between(vel_pred_list_[:, 0, -1], upper_bound, lower_bound, color='b', alpha=.3)

# ax.plot(vel_pred_list_[:, 0, -1], vel_pred_list_[:, 0, 1] , label="x", color="g")
# ax.plot(vel_pred_list_[:, 0, -1], vel_pred_list_[:, 0, 2] , label="x", color="b")

ax.set_title('Learning Based Proprioceptive Odometry')
# ax.set_ylim(-1,1)
ax.set_xlabel("Time")
ax.set_ylabel("Velocity $(m/s)$ in IMU frame")

plt.legend()
plt.grid()
plt.show()
