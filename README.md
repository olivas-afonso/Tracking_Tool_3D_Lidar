# Car Tracking Using 3D LiDAR

This project demonstrates how to track a car using 3D LiDAR data and estimate its 2D ground plane odometry. The implementation leverages ROS2 for data handling and Open3D for point cloud processing.

## Overview

### Tracking a Car with 3D LiDAR

Car tracking involves detecting and following the position of a car in real-time using point cloud data from a 3D LiDAR sensor.

### 2D Ground Plane Odometry

Odometry estimates the position and orientation of the car on the 2D ground plane over time using LiDAR data. By tracking the movement of detected car clusters frame-to-frame, you can compute the trajectory.
## Steps

1. **Data Acquisition Environment Point Cloud**

    Begin by recording a ROS2 bag file that captures the environment using a 3D LiDAR sensor, ensuring that the car to be tracked is not present during this process. This baseline point cloud serves as a reference for distinguishing dynamic objects, such as the car, from static background elements in subsequent tracking steps.
    
    - Clone the `HesaiLidar_general_Ros` repository inside your workspace:
        ```bash      
        git submodule init
        git submodule update
        cd HesaiLidar_general_Ros
        ```
    - Refer to the `README.md` inside `HesaiLidar_general_Ros` for setup and usage instructions.

2. **Preprocessing Environment Point Cloud**

    Run the `save_env.py` script to process the recorded ROS2 bag file. This script extracts and saves the environment point cloud, which will be used as a reference for filtering out static background points during the car tracking process.
    
    ```bash
    python save_env.py /path/to/rosbag
    ```

3. **Object Detection and Tracking**

    Use the `track_ROS_node.py` script to detect and track the car in real-time. This script processes incoming LiDAR data, identifies the car based on its size and shape, and tracks its position over time.
    
    ```bash
    python track_ROS_node.py
    ```

    Key steps in the script include:

    - **Preprocessing**: Filter and downsample the point cloud data.
    - **Segmentation**: Remove ground points to isolate potential objects.
    - **Clustering**: Group remaining points into clusters representing individual objects.
    - **Car Identification**: Select clusters whose dimensions and shape correspond to typical car profiles.

4. **Odometry Calculation**

    Estimate the car's position and orientation on the 2D ground plane. The script invokes `ground_plane_bag.py` to extract the ground plane and perform the necessary transformation for 2D odometry.

    ```bash
    python lidar_odom.py
    ```
