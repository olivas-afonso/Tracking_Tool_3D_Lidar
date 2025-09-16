"""
ground_plane_bag.py

This script loads point cloud data from a ROS2 bag file, fits a ground plane using RANSAC regression,
and saves the plane coefficients. Optionally, it visualizes the point cloud and the fitted plane.

Usage:
    python ground_plane_bag.py <bag_path>

Dependencies:
    - numpy
    - scikit-learn
    - matplotlib
    - ROS2 Python libraries (rclpy, rosbag2_py, sensor_msgs, sensor_msgs_py)
"""
import numpy as np
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import rosbag2_py
import os

def load_point_cloud_from_bag(bag_path, topic, n_msgs):
    rclpy.init()  # Initialize rclpy before using serialization
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    points = []
    msg_type = PointCloud2  # Define the message type for deserialization
    msg_count = 0
    max_msgs = n_msgs
    while reader.has_next() and msg_count < max_msgs:
        (topic_name, data, t) = reader.read_next()
        if topic_name == topic:
            msg = deserialize_message(data, msg_type)
            for p in pc2.read_points(msg, skip_nans=True):
                points.append([p[0], p[1], p[2]])
            msg_count += 1
    rclpy.shutdown()  # Shutdown rclpy after use
    return np.array(points)

# Example usage:

def compute_ground_plane_from_bag(bag_path, ws_folder=None, plot_flag=True, n_msgs=50):
    # Load and filter points
    points = load_point_cloud_from_bag(bag_path, '/pandar', n_msgs=n_msgs)
    
    # Filter for likely ground points
    z_min, z_max = -2.0, 2.0
    height_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    filtered_points = points[height_mask]
    
    # Remove points too close to origin (sensor noise)
    distance = np.linalg.norm(filtered_points[:, :2], axis=1)
    distance_mask = (distance > 1.0) & (distance < 50.0)
    background = filtered_points[distance_mask]
    
    if len(background) < 100:
        print("WARNING: Too few points for reliable plane fitting")
        return np.array([0, 0, 0])
    
    # Fit plane with tuned RANSAC
    X = background[:, :2]
    y = background[:, 2]
    
    ransac = RANSACRegressor(
        min_samples=3,
        residual_threshold=0.03,  # Tighter threshold
        max_trials=2000,
        stop_probability=0.99,
        random_state=42
    )
    
    ransac.fit(X, y)
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    
    # Calculate quality metrics
    inlier_mask = ransac.inlier_mask_
    inliers = background[inlier_mask]
    predicted_z = ransac.predict(inliers[:, :2])
    rmse = np.sqrt(np.mean((inliers[:, 2] - predicted_z)**2))
    
    print(f"RANSAC results:")
    print(f"  Inliers: {np.sum(inlier_mask)}/{len(background)}")
    print(f"  RMSE: {rmse:.4f} m")
    print(f"  Plane: z = {a:.4f}*x + {b:.4f}*y + {c:.4f}")
    
    # Save coefficients
    filename = f"ground_plane_{'background_full'}.npy"
    output_file = os.path.join(ws_folder, filename)
    np.save(output_file, np.array([a, b, c]))
    
    if plot_flag:
        # Enhanced visualization
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 5))
        
        # 3D plot
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(background[:, 0], background[:, 1], background[:, 2], 
                   s=1, c='blue', alpha=0.5, label='All points')
        ax1.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2],
                   s=2, c='green', alpha=0.7, label='Inliers')
        
        # Plot plane
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                            np.linspace(ylim[0], ylim[1], 10))
        zz = a * xx + b * yy + c
        ax1.plot_surface(xx, yy, zz, alpha=0.5, color='red')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # 2D residual plot
        ax2 = fig.add_subplot(122)
        residuals = inliers[:, 2] - predicted_z
        ax2.hist(residuals, bins=50, alpha=0.7)
        ax2.set_xlabel('Residual (m)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Residual Distribution (RMSE: {rmse:.3f}m)')
        
        plt.tight_layout()
        plt.show()
    
    return np.array([a, b, c])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ground_plane_bag.py <bag_path>")
        bag_path = "/home/olivas/camera_ws/bags/rosbag2_2025_09_16-11_50_59"

        #sys.exit(1)
    else:
        bag_path = sys.argv[1]
    
    bag_folder = os.path.normpath(bag_path).split(os.sep)[0]
    print(f"Bag folder: {bag_folder}")
    
    compute_ground_plane_from_bag(bag_path, ws_folder=bag_folder ,plot_flag=True, n_msgs=50)