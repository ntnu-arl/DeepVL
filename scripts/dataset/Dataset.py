import numpy as np
import os
import torch

class dbioDataset(object):
    def __init__(self, dataset_dir, dataset_config = None, traj_name=None, train = True):
        self.dataset_dir = dataset_dir
        self.dataset_config = dataset_config
        
        self.seq_duration = dataset_config["seq_duration"]
        assert dataset_config["seq_sample_density"]< 50
        self.seq_sample_density = dataset_config["seq_sample_density"]
        
        self.alphasense_imu_fps = dataset_config["sensor_fps"]["alphasense_imu"]
        self.fcu_imu_fps = dataset_config["sensor_fps"]["fcu_imu"]
        self.gravity_b_vec_fps = dataset_config["sensor_fps"]["rovio_odom_twist"]
        self.motor_command_fps = dataset_config["sensor_fps"]["motor_command"]
        self.supervision_fps = dataset_config["sensor_fps"]["rovio_odom_twist"]
        self.biases_fps = dataset_config["sensor_fps"]["rovio_odom_twist"]
        self.battery_fps = dataset_config["sensor_fps"]["battery"]
        self.gravity_correction = dataset_config["gravity_correction"]
        self.step = dataset_config["step"]
        self.train = train
        self.traj_name = None

        # normalize data parameters
        self.alphasense_imu_mean = np.array( dataset_config["sensor_stats"]["alphasense_imu"]["mean"]) # time stamp is not included
        self.alphasense_imu_std = np.array(dataset_config["sensor_stats"]["alphasense_imu"]["std"] ) # time stamp is not included

        self.fcu_imu_mean = np.array(dataset_config["sensor_stats"]["fcu_imu"]["mean"] ) # time stamp is not included
        self.fcu_imu_std = np.array(dataset_config["sensor_stats"]["fcu_imu"]["std"]) # time stamp is not included
        
        self.biases_imu_mean = np.array([ 0.0, 0.0, 0.0,  0.0, 0.0,  0.0 ] ) # time stamp is not included
        self.biases_imu_std = np.array([ 0.2, 0.2, 0.2, 0.02, 0.2, 0.2 ] ) # time stamp is not included
        
        self.alphasense_imu_ds_gc_mean = np.array([0.0,  0.0, 0.0,  6.45469411e-04,  4.10618554e-03,  6.86019256e-05 ] ) # time stamp is not included
        self.alphasense_imu_ds_gc_std = np.array(dataset_config["sensor_stats"]["alphasense_imu"]["std"] ) # time stamp is not included
        
        self.motor_command_mean = np.array(dataset_config["sensor_stats"]["motor_command"]["mean"]) # time stamp is not included
        self.motor_command_std = np.array(dataset_config["sensor_stats"]["motor_command"]["std"] ) # time stamp is not included
        
        self.supervision_mean = np.array(dataset_config["sensor_stats"]["rovio_odom_twist"]["mean"]) # time stamp is not included
        self.supervision_std = np.array(dataset_config["sensor_stats"]["rovio_odom_twist"]["std"]) # time stamp is not included
        
        self.battery_mean = np.array(dataset_config["sensor_stats"]["battery"]["mean"]) # time stamp is not included
        self.battery_std = np.array(dataset_config["sensor_stats"]["battery"]["std"]) # time stamp is not included
        
        # load data
        self.data = self.load_data()
        
        self.data_len = len(self.data)
        self.data_idx = np.arange(self.data_len)
    
    def get_fragment(self, fragment_length, data_array, t):
        # returns the start and end indecies of the fragment
        # data_array is a numpy array with the last column being the time stamp
        end_index = np.argmin(np.abs(data_array[:, -1] - t))
        
        time_diff = np.argmin(data_array[end_index, -1] - t)
        if time_diff > 1.0:
            print("Time difference is greater than 1.0")
        
        start_index = end_index-int(fragment_length)
        
        return data_array[start_index:end_index, :], start_index, end_index
    
    def normalize_data(self, data, mean, std):
        # normalize data
        data = data.copy()
        data[:, :-1] = (data[:, :-1] - mean)/std
        return data
    
    def denormalize_data(self, data, mean, std):
        # de normalize data
        data = data.copy()
        data[:, :-1] = (data[:, :-1]*std) + mean
        return data
    
    def sample_fragments(self, start_time= 0, end_time= 0, alphasense_imu_data=None, 
                                                            fcu_imu_data=None,
                                                            motor_command_data=None,
                                                            supervision_data=None,
                                                            gravity_b_vec=None,
                                                            orientation_data_Rmat=None,
                                                            biases_data = None, 
                                                            battery_data = None,
                                                            N=10, 
                                                            sampling_method="random", 
                                                            step = 1):
        
        # sample N points from the time interval [start_time, end_time]
        # return the sampled points as a numpy array
        data = []
        
        N = int(N)
        
        if sampling_method == "random":            
            for i in range(N):
                t = np.random.uniform(start_time, end_time)
                
                fragment_length = self.seq_duration
                
                alphasense_imu_frag, _, _ = self.get_fragment(fragment_length*self.alphasense_imu_fps, alphasense_imu_data, t)
                if gravity_b_vec is not None:
                    gravity_b_vec_frag, _, _ = self.get_fragment(fragment_length*self.gravity_b_vec_fps, gravity_b_vec, t )
                fcu_imu_frag, _, _ = self.get_fragment(fragment_length*self.fcu_imu_fps, fcu_imu_data, t)
                motor_command_frag, _, _ = self.get_fragment(fragment_length*self.motor_command_fps, motor_command_data, t)
                supervision_frag, _, _ = self.get_fragment(fragment_length*self.supervision_fps, supervision_data, t)
                biases_frag, _, _ = self.get_fragment(fragment_length*self.biases_fps, biases_data, t)
                supervision_frag, start_i, end_i = self.get_fragment(fragment_length*self.supervision_fps, supervision_data, t)
                orientation_data_Rmat_frag_w = orientation_data_Rmat[start_i:end_i]
                orientation_data_Rmat_0 = orientation_data_Rmat_frag_w[-1:].transpose(0,2,1) # Inverse of first frame in the sequence
                orientation_data_Rmat_frag = np.matmul(orientation_data_Rmat_frag_w, orientation_data_Rmat_0) # Transforming w.r.t first frame in the sequence
                if battery_data is not None:
                    time_diff = battery_data[-1, -1] - battery_data[-11, -1]
                    inst_battery_fps = 10.0/time_diff

                    valid_fps = np.array([10, 20, 50, 100, 200])
                    inst_battery_fps = valid_fps[np.argmin(np.abs(valid_fps - inst_battery_fps))]
                    
                    battery_frag, _, _ = self.get_fragment(fragment_length*inst_battery_fps, battery_data, t)
                # downsample by taking mean for each 10 points
                
                if alphasense_imu_frag.shape[0]==self.alphasense_imu_fps*fragment_length and \
                    motor_command_frag.shape[0]==self.motor_command_fps*fragment_length and \
                    fcu_imu_frag.shape[0]==self.fcu_imu_fps*fragment_length and \
                    supervision_frag.shape[0]==self.supervision_fps*fragment_length:
                    # Data augmentation                    
                    # vertical thruster noise
                    if self.train:
                        motor_command_frag[:, 0:8] = motor_command_frag[:, 0:8]*(1 + np.random.randn()*self.dataset_config["thruster_scale_noise"]["train"])
                        motor_command_frag[:, 4:8] = motor_command_frag[:, 4:8] + np.random.randn()*self.dataset_config["vertical_thruster_noise"]["train"]
                    else:
                        motor_command_frag[:, 0:8] = motor_command_frag[:, 0:8]*(1 + np.random.randn()*self.dataset_config["thruster_scale_noise"]["test"])                        
                        motor_command_frag[:, 4:8] = motor_command_frag[:, 4:8] + np.random.randn()*self.dataset_config["vertical_thruster_noise"]["test"]

                    alphasense_imu_down_sampled = np.mean(alphasense_imu_frag.reshape(-1, 10, 7), axis=1)

                    G_MAG = 9.8065
                    if gravity_b_vec is not None:
                        alphasense_imu_ds_gc = alphasense_imu_down_sampled.copy() # gravity corrected
                        
                        alphasense_imu_ds_gc[:, :3] = alphasense_imu_ds_gc[:, :3] - (gravity_b_vec_frag[:, :3])
                        
                        alphasense_imu_ds_gc[:, 3:6] = alphasense_imu_ds_gc[:, 3:6] - biases_frag[:, 3:6]

                    # normalize data
                    alphasense_imu_ds_normed = self.normalize_data(alphasense_imu_down_sampled, self.alphasense_imu_mean, self.alphasense_imu_std)
                    motor_command_normed = self.normalize_data(motor_command_frag, self.motor_command_mean, self.motor_command_std)
                    
                    if battery_data is not None:
                        battery_frag_normed = self.normalize_data(battery_frag, self.battery_mean, self.battery_std)                    
                        
                        # downsample battery data from inst_battery_fps to 10 fps
                        if inst_battery_fps == 20:
                            battery_frag_normed = battery_frag_normed[::2]
                        elif inst_battery_fps == 50:
                            battery_frag_normed = battery_frag_normed[::5]
                            
                        # upsample battery data to 20 fps
                        battery_frag_normed_up = np.repeat(battery_frag_normed, 2, axis=0)
                        
                        battery_mask = np.ones((battery_frag_normed_up.shape[0], 1))
                        battery_frag_normed_up = np.concatenate((battery_mask, battery_frag_normed_up), axis=1)
                    else:
                        # dummy battery data zeros with zero mask
                        battery_frag_normed_up = np.zeros((int(self.battery_fps*fragment_length), 3))

                    supervision_normed = self.normalize_data(supervision_frag, self.supervision_mean, self.supervision_std)
                    
                    # double to float
                    alphasense_imu_ds_normed = alphasense_imu_ds_normed.astype(np.float32)
                    motor_command_normed = motor_command_normed.astype(np.float32)
                    supervision_normed = supervision_normed.astype(np.float32)
                    battery_frag_normed_up = battery_frag_normed_up.astype(np.float32)
                    
                    # downsample
                    ds_factor = self.dataset_config["downsample_factor"]
                    data.append({"alphasense_imu_ds_normed": alphasense_imu_ds_normed,
                                "motor_command_normed": motor_command_normed[::ds_factor],
                                # "supervision": supervision_frag,
                                "supervision_normed": supervision_normed,
                                "battery_voltage": battery_frag_normed_up})
                        
        return data
                        
    def load_data(self):
        traj_dirs = os.listdir(self.dataset_dir)
        traj_dirs.sort()
        data = []

        if self.traj_name is None:
            for traj_dir in traj_dirs:
                traj_dir = os.path.join(self.dataset_dir, traj_dir)

                if os.path.isdir(traj_dir):                
                    
                    alphasense_imu_data = np.load(os.path.join(traj_dir, "alphasense_imu_data.npy"))
                    fcu_imu_data = np.load(os.path.join(traj_dir, "fcu_imu_data.npy"))
                    if self.gravity_correction:
                        gravity_b_vec = np.load(os.path.join(traj_dir, "gravity_b_vec.npy"))
                    else:
                        gravity_b_vec = None

                    # orientation_data_Rmat = np.load(os.path.join(traj_dir, "orientation_data_Rmat.npy"))                    
                    motor_command_data = np.load(os.path.join(traj_dir, "motor_commands_data.npy"))
                    supervision_data = np.load(os.path.join(traj_dir, "supervision_odom_data.npy"))
                    biases_data = np.load(os.path.join(traj_dir, "biases_data.npy"))
                    orientation_data_Rmat = np.load(os.path.join(traj_dir, "orientation_data_Rmat.npy"))
                    
                    # check if battery file exists
                    if os.path.isfile(os.path.join(traj_dir, "battery_data.npy")):
                        battery_data = np.load(os.path.join(traj_dir, "battery_data.npy"))
                        print("battery data loaded")
                    else:
                        battery_data = None
                        print("battery data not loaded")

                    start_time = supervision_data[0, -1]
                    end_time = supervision_data[-1, -1]
                    
                    print("Time difference", end_time - start_time)
                    
                    data.extend(self.sample_fragments(start_time=start_time, 
                                                    end_time=end_time, 
                                                    alphasense_imu_data=alphasense_imu_data, 
                                                    fcu_imu_data=fcu_imu_data, 
                                                    motor_command_data=motor_command_data, 
                                                    supervision_data=supervision_data,
                                                    orientation_data_Rmat=orientation_data_Rmat,
                                                    gravity_b_vec=gravity_b_vec,
                                                    biases_data = biases_data,
                                                    battery_data=battery_data, 
                                                    N = (end_time - start_time)*self.seq_sample_density))
            return data
                    
        else:
            
            traj_dir = os.path.join(self.dataset_dir, self.traj_name)
            print("traj_dir", traj_dir)
            if os.path.isdir(traj_dir):                
                alphasense_imu_data = np.load(os.path.join(traj_dir, "alphasense_imu_data.npy"))
                fcu_imu_data = np.load(os.path.join(traj_dir, "fcu_imu_data.npy"))
                if self.gravity_correction:
                        gravity_b_vec = np.load(os.path.join(traj_dir, "gravity_b_vec.npy"))
                else:
                    gravity_b_vec = None
                # orientation_data_Rmat = np.load("orientation_data_Rmat.npy")
                
                motor_command_data = np.load(os.path.join(traj_dir, "motor_commands_data.npy"))
                supervision_data = np.load(os.path.join(traj_dir, "supervision_odom_data.npy"))
                
                self.test_data = {"alphasense_imu": alphasense_imu_data,
                                    "motor_command": motor_command_data, 
                                    "supervision": supervision_data}
                
                data = None                
                
            else:
                print ("Incorrect path: ", traj_dir)
        

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data_idx)

if __name__=="__main__":
    dataset_dir = "/home/sakura/ssd/Datasets/MClabDataset/DBIO/test_gravity"
    dataset = dbioDataset(dataset_dir)
    # print(dataset[10])
    print(len(dataset))
    print(dataset[0]["alphasense_imu"].shape)
    print(dataset[0]["alphasense_imu_ds"].shape)
    print("alphasense_imu_ds_gc_normed", dataset[0]["alphasense_imu_ds_gc_normed"].shape)
    # print(dataset[0]["fcu_imu"].shape)
    print(dataset[0]["motor_command"].shape)
    print(dataset[0]["supervision"].shape)