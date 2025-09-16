"""
This script implements a ROS2 node for 3D LiDAR-based moving object detection using background subtraction and clustering.

Main features:
- Loads a precomputed static background point cloud.
- Subscribes to incoming LiDAR point clouds.
- Performs background subtraction using a KD-tree for fast nearest neighbor search.
- Clusters moving points using DBSCAN to identify distinct moving objects.
- Publishes:
    - PointCloud2 of moving objects.
    - MarkerArray for bounding boxes visualization.
    - Odometry messages for detected object centroids.
    
Usage:
    python track_ROS_node.py
Requirements:
    - A precomputed background point cloud saved as a NumPy array like 'background_full.npy'.
    - ROS2 environment with necessary packages installed.
Arguments:
    None
Outputs:
    - Published topics for moving objects, bounding boxes, and odometry.

"""


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import numpy as np

class BackgroundSubtractor(Node):
    def __init__(self):
        super().__init__('background_subtractor')

        self.get_logger().info("A carregar ambiente de fundo...")
        self.bg = np.load('/home/olivas/camera_ws/bags/background_full.npy')

        if self.bg.dtype.names:
            self.bg = np.stack([self.bg['x'], self.bg['y'], self.bg['z']], axis=-1)
        self.bg = self.bg.astype(np.float32)

        self.kdtree = cKDTree(self.bg)

        self.sub = self.create_subscription(PointCloud2, '/pandar', self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, '/moving_objects_env', 10)
        self.bbox_pub = self.create_publisher(MarkerArray, '/moving_objects_bboxes', 10)
        self.odom_pub = self.create_publisher(Odometry, '/moving_objects_odom', 10)

    def make_field(self, name, offset):
        return PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)

    def callback(self, msg):
        points = np.array([ [p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True) ], dtype=np.float32)

        if points.shape[0] == 0:
            self.get_logger().info("Nenhum ponto recebido.")
            return

        # Subtração de fundo
        dists, _ = self.kdtree.query(points, k=1, workers=-1)
        moving_mask = dists > 0.08
        moving = points[moving_mask]

        if moving.shape[0] == 0:
            return

        # Clustering
        clustering = DBSCAN(eps=0.8, min_samples=5, n_jobs=-1).fit(moving)
        labels = clustering.labels_

        if np.all(labels == -1):
            return

        # Publicar nuvem de pontos dos objetos em movimento
        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        fields = [self.make_field('x', 0), self.make_field('y', 4), self.make_field('z', 8)]
        pc2_msg = pc2.create_cloud(header, fields, moving[labels != -1])
        self.pub.publish(pc2_msg)

        # Bounding boxes e odometria
        marker_array = MarkerArray()
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_pts = moving[labels == cluster_id]
            centroid = np.mean(cluster_pts, axis=0)
            min_xyz = np.min(cluster_pts, axis=0)
            max_xyz = np.max(cluster_pts, axis=0)

            marker = Marker()
            marker.header = header
            marker.ns = "bbox"
            marker.id = int(cluster_id)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float((min_xyz[0] + max_xyz[0]) / 2)
            marker.pose.position.y = float((min_xyz[1] + max_xyz[1]) / 2)
            marker.pose.position.z = float((min_xyz[2] + max_xyz[2]) / 2)
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(max_xyz[0] - min_xyz[0])
            marker.scale.y = float(max_xyz[1] - min_xyz[1])
            marker.scale.z = float(max_xyz[2] - min_xyz[2])
            marker.color.r = 1.0
            marker.color.a = 0.5
            marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
            marker_array.markers.append(marker)

            odom = Odometry()
            odom.header = header
            odom.child_frame_id = f"object_{cluster_id}"
            odom.pose.pose.position.x = float(centroid[0])
            odom.pose.pose.position.y = float(centroid[1])
            odom.pose.pose.position.z = float(centroid[2])
            odom.pose.pose.orientation.w = 1.0
            self.odom_pub.publish(odom)

        self.bbox_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = BackgroundSubtractor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
