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

def compute_ground_plane_from_bag(bag_path,ws_folder=None, plot_flag=True, n_msgs=50):


    # Load the background image (assumed to be a depth or point cloud map)
    points= load_point_cloud_from_bag(bag_path, '/pandar', n_msgs=n_msgs)



    background = points

    # Example: Compute the ground plane using RANSAC (assuming background is Nx3 points)

    # Assume background is (N, 3): x, y, z
    X = background[:, :2]  # x, y
    y = background[:, 2]   # z

    # Fit plane: z = a*x + b*y + c
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    print(f"Ground plane equation: z = {a:.4f}*x + {b:.4f}*y + {c:.4f}")

    # Save the ground plane coefficients to a file
    
    filename = f"ground_plane_{'background_full'}.npy"
    output_file=os.path.join(ws_folder, filename)
    print(f"Saving ground plane coefficients to {output_file}")
    np.save(output_file, np.array([a, b, c]))
    
    
    if plot_flag:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the point cloud
        ax.scatter(background[:, 0], background[:, 1], background[:, 2], s=1, c='b', label='Points')

        # Create a meshgrid for the plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx, yy = np.meshgrid(
            np.linspace(xlim[0], xlim[1], 10),
            np.linspace(ylim[0], ylim[1], 10)
        )
        zz = a * xx + b * yy + c

        # Plot the plane
        ax.plot_surface(xx, yy, zz, alpha=0.5, color='r', label='Fitted Plane')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #plt.legend()
        plt.show()
        
    return  np.array([a, b, c])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ground_plane_bag.py <bag_path>")
        bag_path = "track_tools/test_06_23/test_12_08/rosbag2_2025_06_23-12_08_31_lidar"

        #sys.exit(1)
    else:
        bag_path = sys.argv[1]
    
    bag_folder = os.path.normpath(bag_path).split(os.sep)[0]
    print(f"Bag folder: {bag_folder}")
    
    compute_ground_plane_from_bag(bag_path, ws_folder=bag_folder ,plot_flag=True, n_msgs=50)