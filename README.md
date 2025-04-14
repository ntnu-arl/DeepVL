# <div align="center">DeepVL: Deep Velocity Learning </div>

<div align="center"> <a href="https://ntnu-arl.github.io/deepvl-deep-velocity-learning/"><img src="https://img.shields.io/badge/Homepage-1E88E5?style=flat-square" alt="Homepage"></a> <a href="https://arxiv.org/abs/2502.07726v1"><img src="https://img.shields.io/badge/arXiv-78909C?style=flat-square" alt="arXiv"></a> <a href="https://youtu.be/ctcbrNu_N78?feature=shared"><img src="https://img.shields.io/badge/YouTube-E57373?style=flat-square" alt="YouTube"></a> </div>

**DeepVL** is an open-source framework for predicting linear velocity of underwater robots, enabling robust state estimation during blackouts of exteroceptive sensors (e.g., cameras, DVLs, or UBSLs). It takes the actuator commands, IMU measurements, and battery voltage as an input and predicts the robot velocity in the body frame.

<div align="center">
  <img src="media/Teaser.png" alt="DeepVL Teaser" width="750"/>
</div>

This repository contains the code for the paper:  
**"DeepVL: Dynamics and Inertial Measurements-based Deep Velocity Learning for Underwater Odometry"**  
*Mohit Singh, Kostas Alexis* | [arXiv:2502.07726](https://arxiv.org/abs/2502.07726)

---

## 🌟 Features
- Predicts linear velocity using inertial and actuator data
- Handles sensor blackouts for reliable underwater state-estimation
- Integrates with Extended Kalman Filters (e.g., ReAqROVIO)
- Pre-trained models and ROS node for quick deployment

---

## Quick Start

### 1. Installation
Clone the repository and set up the environment:

```bash
git clone https://github.com/ntnu-arl/deepvl.git
cd deepvl
conda env create -f environment.yml
conda activate deepvl
```

### 2. Inference Dataset Setup
Download the dataset in ROS bag format from [link (inference dataset)](https://huggingface.co/datasets/ntnu-arl/underwater-datasets). Update the dataset path in the configuration YAML:

```yaml
dataset:
    dataset_dir: <path-to-dataset>
```

### 3. Running DeepVL
Launch the ROS node with pre-trained weights:

```bash
roslaunch deepvl_ros inertial_odom_ros.launch
```

The estimated velocity and covariance are published to:

```
/deepvl_vel_cov/twist
```

---

## Usage

### Integration with ReAqROVIO
DeepVL predictions can be fused into the Refractive Aquatic ROVIO (ReAqROVIO) framework for enhanced underwater state estimation. Remap the velocity and covariance outputs to your EKF pipeline, combining with barometric and inertial data as needed.

### Training Your Model
To train the DeepVL ensemble:

1. Download the dataset with train and test data in `.npy` format from [link (training dataset)](https://huggingface.co/datasets/ntnu-arl/deepvl-training-data).
2. Run the training script:

```bash
python3 train_inertial_odom.py
```

Customize training via the configuration YAML file, where all parameters are documented as comments.

---

## Citation
If you use DeepVL in your research, please cite:

```bibtex
@misc{singh2025deepvldynamicsinertialmeasurementsbased,
  title={DeepVL: Dynamics and Inertial Measurements-based Deep Velocity Learning for Underwater Odometry},
  author={Mohit Singh and Kostas Alexis},
  year={2025},
  eprint={2502.07726},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2502.07726},
}
```

---

## Contact
For questions or support, reach out via [GitHub Issues](https://github.com/ntnu-arl/deepvl/issues) or contact authors:

* [Mohit Singh](mailto:mohit.singh@ntnu.no)
* [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)

---
