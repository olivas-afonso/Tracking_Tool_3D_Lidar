"""
save_env.py

This script loads point cloud data from a ROS2 bag file, accumulates all points from the '/pandar' topic,
and saves the aggregated environment as both .pcd and .npy files. Optionally, it applies voxel downsampling
to reduce point cloud density.

Usage:
    python save_env.py

Dependencies:
    - numpy
    - open3d
    - ROS2 Python libraries (rclpy, rosbag2_py, sensor_msgs, sensor_msgs_py)

Returns:
    - Aggregated environment point cloud saved as 'background_full.pcd' and 'background_full.npy' in the workspace folder.
"""
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import os
import sys

rclpy.init()

if len(sys.argv) > 1:
    bag_path = sys.argv[1]
else:
    bag_path = './track_tools/test_06_23/env/rosbag2_2025_06_23-11_51_44'

# Configurar leitura do bag
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions('cdr', 'cdr')
reader.open(storage_options, converter_options)
reader.set_filter(rosbag2_py.StorageFilter(topics=['/pandar']))
msg_type = get_message('sensor_msgs/msg/PointCloud2')

# Acumular pontos
all_points = []

i = 0
while reader.has_next():
    topic, data, t = reader.read_next()
    msg = deserialize_message(data, msg_type)
    points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

    if points.shape[0] == 0:
        continue
    all_points.append(points)
        
    i += 1

print(f"[INFO] Processadas {i} mensagens...")

# Concatenar todos os pontos
background = np.concatenate(all_points, axis=0)
xyz_array = np.stack([background['x'], background['y'], background['z']], axis=-1).astype(np.float32)
print(f"[✔] Total de {background.shape[0]} pontos acumulados.")

# Converter para nuvem Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_array)

# (Opcional) Aplicar voxel downsampling para reduzir densidade
pcd = pcd.voxel_down_sample(voxel_size=0.05)

output_dir = os.path.dirname(storage_options.uri)
# Salvar como .pcd e .npy
o3d.io.write_point_cloud(os.path.join(output_dir, "background_full.pcd"), pcd)

np.save(os.path.join(output_dir, "background_full.npy"), np.asarray(pcd.points))
print(f"[✔] Ambiente salvo como 'background_full.pcd' e 'background_full.npy on {output_dir}'.")


