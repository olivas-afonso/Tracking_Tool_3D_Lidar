"""
ground_plane_bag_enhanced.py

Enhanced version for better ground plane fitting using RANSAC with stricter parameters.
This version aims for higher precision at the cost of longer computation time.

Usage:
    python ground_plane_bag_enhanced.py <bag_path>

Dependencies:
    - numpy
    - scikit-learn
    - matplotlib
    - ROS2 Python libraries (rclpy, rosbag2_py, sensor_msgs, sensor_msgs_py)
"""
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import rosbag2_py
import os
import time

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

def compute_ground_plane_from_bag(bag_path, plot_flag=True, n_msgs=50):
    start_time = time.time()
    
    # Load and filter points
    points = load_point_cloud_from_bag(bag_path, '/pandar', n_msgs=n_msgs)
    print(f"Loaded {len(points)} points from bag file")
    
    # More selective filtering for ground points
    z_min, z_max = -1.5, 0.5  # Tighter vertical range for ground points
    height_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    filtered_points = points[height_mask]
    print(f"After height filtering: {len(filtered_points)} points")
    
    # Remove points too close to origin (sensor noise) and too far
    distance = np.linalg.norm(filtered_points[:, :2], axis=1)
    distance_mask = (distance > 1.5) & (distance < 40.0)  # Adjusted range
    background = filtered_points[distance_mask]
    print(f"After distance filtering: {len(background)} points")
    
    if len(background) < 100:
        print("WARNING: Too few points for reliable plane fitting")
        return np.array([0, 0, 0])
    
    # Fit plane with enhanced RANSAC parameters
    X = background[:, :2]
    y = background[:, 2]
    
    # Run multiple RANSAC iterations to find the best fit
    best_inliers = 0
    best_rmse = float('inf')
    best_coeffs = None
    best_inlier_mask = None
    
    n_iterations = 5  # Run multiple times to find best fit
    for i in range(n_iterations):
        print(f"RANSAC iteration {i+1}/{n_iterations}")
        
        ransac = RANSACRegressor(
            min_samples=3,
            residual_threshold=0.02,  # Tighter threshold for better precision
            max_trials=5000,          # More trials for better chance of good fit
            stop_probability=0.995,   # Higher probability for more exhaustive search
            random_state=42+i         # Different seed each iteration
        )
        
        ransac.fit(X, y)
        a, b = ransac.estimator_.coef_
        c = ransac.estimator_.intercept_
        
        # Calculate quality metrics
        inlier_mask = ransac.inlier_mask_
        n_inliers = np.sum(inlier_mask)
        inliers = background[inlier_mask]
        predicted_z = ransac.predict(inliers[:, :2])
        rmse = np.sqrt(np.mean((inliers[:, 2] - predicted_z)**2))
        
        # Check if this is the best iteration so far
        if n_inliers > best_inliers or (n_inliers == best_inliers and rmse < best_rmse):
            best_inliers = n_inliers
            best_rmse = rmse
            best_coeffs = np.array([a, b, c])
            best_inlier_mask = inlier_mask
    
    # Use the best result
    a, b, c = best_coeffs
    inlier_mask = best_inlier_mask
    inliers = background[inlier_mask]
    predicted_z = a * inliers[:, 0] + b * inliers[:, 1] + c
    
    # Calculate plane normal and check if it's roughly horizontal
    normal = np.array([-a, -b, 1])
    normal = normal / np.linalg.norm(normal)
    vertical_component = abs(normal[2])  # Should be close to 1 for horizontal plane
    
    print(f"\nRANSAC results after {n_iterations} iterations:")
    print(f"  Inliers: {best_inliers}/{len(background)} ({best_inliers/len(background)*100:.1f}%)")
    print(f"  RMSE: {best_rmse:.6f} m")
    print(f"  Plane: z = {a:.6f}*x + {b:.6f}*y + {c:.6f}")
    print(f"  Plane normal: {normal}")
    print(f"  Vertical component: {vertical_component:.3f}")
    
    # Check if the plane is roughly horizontal
    if vertical_component < 0.9:
        print("WARNING: Fitted plane may not be horizontal (vertical component < 0.9)")
    
    # Save coefficients in current working directory
    filename = "ground_plane_enhanced.npy"
    output_file = os.path.join(os.getcwd(), filename)
    np.save(output_file, best_coeffs)
    print(f"Saved coefficients to {output_file}")
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    
    if plot_flag:
        # Enhanced visualization
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(background[:, 0], background[:, 1], background[:, 2], 
                   s=1, c='blue', alpha=0.3, label='All points')
        ax1.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2],
                   s=2, c='green', alpha=0.7, label='Inliers')
        
        # Plot plane
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 20),
                            np.linspace(ylim[0], ylim[1], 20))
        zz = a * xx + b * yy + c
        ax1.plot_surface(xx, yy, zz, alpha=0.5, color='red')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Point Cloud with Fitted Plane')
        ax1.legend()
        
        # 2D residual plot
        ax2 = fig.add_subplot(222)
        residuals = inliers[:, 2] - predicted_z
        ax2.hist(residuals, bins=50, alpha=0.7, color='green')
        ax2.set_xlabel('Residual (m)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Residual Distribution (RMSE: {best_rmse:.4f}m)')
        
        # Top-down view (X-Y plane)
        ax3 = fig.add_subplot(223)
        ax3.scatter(background[:, 0], background[:, 1], s=1, c='blue', alpha=0.3, label='All points')
        ax3.scatter(inliers[:, 0], inliers[:, 1], s=2, c='green', alpha=0.7, label='Inliers')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Top-Down View (X-Y Plane)')
        ax3.legend()
        ax3.axis('equal')
        
        # Side view (X-Z plane)
        ax4 = fig.add_subplot(224)
        ax4.scatter(background[:, 0], background[:, 2], s=1, c='blue', alpha=0.3, label='All points')
        ax4.scatter(inliers[:, 0], inliers[:, 2], s=2, c='green', alpha=0.7, label='Inliers')
        
        # Plot plane cross-section
        x_range = np.linspace(ax4.get_xlim()[0], ax4.get_xlim()[1], 100)
        z_plane = a * x_range + c  # Assuming y=0 for simplicity
        ax4.plot(x_range, z_plane, 'r-', linewidth=2, label='Plane cross-section')
        
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        ax4.set_title('Side View (X-Z Plane)')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
    
    return best_coeffs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ground_plane_bag_enhanced.py <bag_path>")
        bag_path = "/home/olivas/camera_ws/bags/rosbag2_2025_09_16-11_50_59"
        # sys.exit(1)
    else:
        bag_path = sys.argv[1]
    
    compute_ground_plane_from_bag(bag_path, plot_flag=True, n_msgs=50)