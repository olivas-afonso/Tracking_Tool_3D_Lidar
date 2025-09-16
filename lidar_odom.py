"""
lidar_odom.py
This script processes odometry data from a ROS2 bag file containing 3D Lidar-based odometry messages. It extracts relative positions and orientations, applies ground plane corrections, and visualizes the trajectory in 2D and 3D plots. The script also filters out erroneous data points based on velocity thresholds and smooths the trajectory using a Savitzky-Golay filter. Outputs include trajectory plots (PNG), an animated GIF of the XY trajectory, and NumPy arrays of the processed and smoothed odometry data.
Main functionalities:
- Reads odometry messages from a specified ROS2 bag file.
- Computes and applies ground plane correction to the trajectory.
- Calculates relative positions and orientations with respect to the initial pose.
- Filters out outlier points based on derivative thresholds.
- Visualizes the trajectory in 3D and 2D (XY plane), saving plots to disk.
- Optionally creates an animated GIF of the XY trajectory.
- Smooths the trajectory using a Savitzky-Golay filter and saves the result.
- Stores processed odometry data as NumPy arrays for further analysis.
Usage:
    python lidar_odom.py /path/to/rosbag
Requirements:
    - A ROS2 bag file containing 3 odometry messages on the specified topic by track ros node.
Arguments:
    bag_path (str): Path to the ROS2 bag file containing odometry data.
Outputs:
    - PNG images of the trajectory (3D, 3D with equal axes, XY plane, smoothed XY).
    - GIF animation of the XY trajectory (optional).
    - NumPy arrays of processed and smoothed odometry data.
Dependencies:
    numpy, matplotlib, scipy, rclpy, rosbag2_py, transforms3d, ground_plane_bag
"""

import os
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from transforms3d.quaternions import qinverse as quaternion_inverse, qmult as quaternion_multiply
from scipy.spatial.transform import Rotation as R
from ground_plane_bag import compute_ground_plane_from_bag

def get_derivative(data, dt=1.0):
    """Compute the derivative of z with respect to time."""
    d = np.diff(data)  # Assuming z is the third column
    return d / dt


def extract_odom_from_rosbag(bag_path, topic_name="/moving_objects_odom", ground_plane=None,ws_folder=None):
    # Setup ROS bag reader
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    # Get topic type for odometry
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    msg_type = get_message(type_map[topic_name])

    odom_data = []
    timestamps = []  
    
    start_pose = None

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == topic_name:
            msg = deserialize_message(data, msg_type)
            if start_pose is None:
                start_pose = msg.pose.pose
                print(f"[INFO] Posição inicial: {start_pose.position.x}, {start_pose.position.y}, {start_pose.position.z}")
                
                
                # do ground plane adjustment
                # Adjust the x, y, z coordinates based on the ground plane
                # Assuming ground_plane is a numpy array with shape (3,) representing the plane coefficients
                a, b, c = ground_plane
                normal = np.array([a, b, -1])  # Normal vector of the ground plane
                normal /= np.linalg.norm(normal)  # Normalize the normal vector
                
                target = np.array([0, 0, 1])
                
                axis = np.cross(normal, target)
                axis_norm = np.linalg.norm(axis)

                if axis_norm > 1e-6:
                    axis = axis / axis_norm
                    angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
                    r_correction_plane = R.from_rotvec(angle * axis)
                    print(f"[INFO] Matriz de rotação calculada: {r_correction_plane.as_matrix()}")
                    print(f"[INFO] Correçao de rotação (Euler): roll={r_correction_plane.as_euler('xyz', degrees=True)[0]:.2f}°, pitch={r_correction_plane.as_euler('xyz', degrees=True)[1]:.2f}°, yaw={r_correction_plane.as_euler('xyz', degrees=True)[2]:.2f}°")
                else:
                    r_correction = R.identity()
                    print("[INFO] Normal vector is too close to the target, using identity rotation.")

                # The car is assumed to be on the ground plane, z_car = 0, 
                q_start = [
                start_pose.orientation.x,
                start_pose.orientation.y,
                start_pose.orientation.z,
                start_pose.orientation.w
                ]
                
                # Correct the start orientation based on the ground plane
               
                # Apply the ground plane correction to the start orientation
                q_start_rot = R.from_quat(q_start)
                q_start_corr = (r_correction_plane * q_start_rot).as_quat()
                print(f"[INFO] Orientação inicial with plane corr: {q_start_corr[0]}, {q_start_corr[1]}, {q_start_corr[2]}, {q_start_corr[3]}")
                print(f"[INFO] Orientação inicial with plane corr (Euler): roll={R.from_quat(q_start_corr).as_euler('xyz', degrees=True)[0]:.2f}°, pitch={R.from_quat(q_start_corr).as_euler('xyz', degrees=True)[1]:.2f}°, yaw={R.from_quat(q_start_corr).as_euler('xyz', degrees=True)[2]:.2f}°")
                
                # Calcule a matriz de rotação para anular o yaw da orientação inicial corrigida
                yaw_angle = R.from_quat(q_start_corr).as_euler('xyz')[2]
                yaw_zero_rot = R.from_euler('z', -yaw_angle)
                rot_matrix_yaw_zero = yaw_zero_rot.as_matrix()
                print(f"[INFO] Matriz de rotação para yaw=0:\n{rot_matrix_yaw_zero}")
                print(f"[INFO] Rotação yaw zero (Euler): roll={yaw_zero_rot.as_euler('xyz', degrees=True)[0]:.2f}°, pitch={yaw_zero_rot.as_euler('xyz', degrees=True)[1]:.2f}°, yaw={yaw_zero_rot.as_euler('xyz', degrees=True)[2]:.2f}°")
                
                r_correction =  yaw_zero_rot * r_correction_plane
                
                
                # Atualize a orientação inicial corrigida aplicando a rotação de yaw_zero_rot
                #
                q_start_corr_rot = R.from_quat(q_start_corr)
                #q_start_corr = (yaw_zero_rot * q_start_corr_rot).as_quat()
                
                q_start_corr = (r_correction * q_start_rot).as_quat()
                
                print(f"[INFO] Orientação inicial corrigida após yaw_zero_rot: {q_start_corr}")
                print(f"[INFO] Orientação inicial corrigida (Euler): roll={R.from_quat(q_start_corr).as_euler('xyz', degrees=True)[0]:.2f}°, pitch={R.from_quat(q_start_corr).as_euler('xyz', degrees=True)[1]:.2f}°, yaw={R.from_quat(q_start_corr).as_euler('xyz', degrees=True)[2]:.2f}°")
                
                
                
              
                
                
                
            """  # Only keep messages within the first 50 seconds
            if len(odom_data) == 0:
                t0 = t
            if (t - t0) / 1e9 > 50:
                break  """
                
            
                
                
            # Compute relative position
            rel_x = msg.pose.pose.position.x - start_pose.position.x
            rel_y = msg.pose.pose.position.y - start_pose.position.y
            rel_z = msg.pose.pose.position.z - start_pose.position.z
            
            
            # Correct the relative position based on the ground plane
            
            # Apply the ground plane correction
            rel_x, rel_y, rel_z = r_correction.apply([rel_x, rel_y, rel_z])
            
           
            # Compute relative orientation (quaternion multiplication: q_rel = q_curr * q_start_inv)
            q_curr = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            
            
            q_curr_rot = R.from_quat(q_curr)
            q_corr = r_correction * q_curr_rot
            q_rel = q_corr.as_quat()  # Convert back to quaternion format
            
            
            
            

            odom_data.append([
                rel_x,
                rel_y,
                rel_z,
                q_rel[0],
                q_rel[1],
                q_rel[2],
                q_rel[3]
            ])
            timestamps.append(t) 
        
    
    import matplotlib.pyplot as plt

    # NEW FILTERING CODE (REPLACE THE ABOVE SECTION):
    odom_data_np = np.array(odom_data)

    # Store timestamps for proper derivative calculation
    #timestamps = []  # You'll need to collect timestamps in the main loop
    # Add this line inside your while loop where you process messages:
    # timestamps.append(t)  # Add this after line where you append to odom_data

    # For now, if you don't have timestamps, we'll use index-based filtering
    # Calculate derivatives with proper indexing
    dz_dt = np.abs(np.diff(odom_data_np[:, 2]))
    dy_dt = np.abs(np.diff(odom_data_np[:, 1]))
    dx_dt = np.abs(np.diff(odom_data_np[:, 0]))

    # Create mask for valid points
    mask = np.ones(len(odom_data_np), dtype=bool)

    # Mark points where derivatives exceed thresholds (starting from index 1)
    for i in range(1, len(odom_data_np)):
        if i-1 < len(dz_dt):  # Ensure we don't go out of bounds
            if (dz_dt[i-1] > 0.1 or dy_dt[i-1] > 0.5 or dx_dt[i-1] > 0.5):
                mask[i] = False

    # Apply the mask
    # Apply the mask
    odom_data_np = odom_data_np[mask]

    print(f"[INFO] Filtered {np.sum(~mask)}/{len(odom_data)} points due to high velocity")

    # Inverte o eixo Y
    odom_data_np[:, 1] = -odom_data_np[:, 1]

    # Calculate new derivatives for plotting (after filtering)
    dz_dt_filtered = np.abs(np.diff(odom_data_np[:, 2]))
    dy_dt_filtered = np.abs(np.diff(odom_data_np[:, 1])) 
    dx_dt_filtered = np.abs(np.diff(odom_data_np[:, 0]))

    deriv_time = np.arange(1, len(dz_dt_filtered) + 1)  # Time indices for the derivative plot

    plt.figure(figsize=(8, 5))
    plt.plot(deriv_time, dx_dt_filtered, label="dz/dt")
    plt.title("Derivada de z entre pontos consecutivos (após filtro)")
    plt.xlabel("Índice da amostra")
    plt.ylabel("dz/dt (metros por amostra)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(odom_data_np[:, 0], odom_data_np[:, 1], odom_data_np[:, 2], label='Trajectory')
    ax.set_title('Relative Odometry Trajectory based on Static Lidar (3D)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    plt.tight_layout()
    plot3d_file = os.path.join(ws_folder, "hesai_odom_trajectory_3d.png")
    plt.savefig(plot3d_file, dpi=300)
    plt.show()
    print(f"[✔] 3D plot saved at '{plot3d_file}'")

    # Save another image with equal aspect ratio for all axes
    fig_eq = plt.figure(figsize=(10, 8))
    ax_eq = fig_eq.add_subplot(111, projection='3d')
    ax_eq.plot(odom_data_np[:, 0], odom_data_np[:, 1], odom_data_np[:, 2], label='Trajectory')
    ax_eq.set_title('Relative Odometry Trajectory (3D, axis equal)')
    ax_eq.set_xlabel('X (meters)')
    ax_eq.set_ylabel('Y (meters)')
    ax_eq.set_zlabel('Z (meters)')
    ax_eq.legend()

    # Set equal aspect ratio for 3D axes
    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max([x_range, y_range, z_range])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    set_axes_equal(ax_eq)
    plt.tight_layout()
    plot3d_equal_file = os.path.join(ws_folder, "hesai_odom_trajectory_3d_equal.png")
    plt.savefig(plot3d_equal_file, dpi=300)
    plt.show()
    print(f"[✔] 3D plot with axis equal saved at '{plot3d_equal_file}'")
    
    if False:

        # Plot only x and y plane and save as GIF
        import matplotlib.animation as animation

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.set_title('Relative Odometry Trajectory based on Static Lidar (XY Plane)')
        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
        ax2.axis('equal')
        line, = ax2.plot([], [], 'b-', label='Trajectory (XY)')
        point, = ax2.plot([], [], 'ro')
        ax2.legend()
        plt.tight_layout()

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate(i):
            line.set_data(odom_data_np[:i+1, 0], odom_data_np[:i+1, 1])
            point.set_data(odom_data_np[i, 0], odom_data_np[i, 1])
            # Ajusta os limites do gráfico dinamicamente para manter a trajetória visível
            ax2.relim()
            ax2.autoscale_view()
            # Fix aspect ratio to be equal (same scale for x and y)
            ax2.set_aspect('equal', adjustable='box')
            return line, point

        ani = animation.FuncAnimation(
            fig2, animate, frames=len(odom_data_np), init_func=init,
            interval=30, blit=True, repeat=False
        )

        gif_file = os.path.join(ws_folder, "hesai_odom_trajectory_xy.gif")
        ani.save(gif_file, writer='pillow', fps=30)
        print(f"[✔] GIF salvo em '{gif_file}'")
        plt.show()
    
    # Plot x and y on the XY plane (static plot)
    plt.figure(figsize=(8, 6))
    plt.plot(odom_data_np[:, 0], odom_data_np[:, 1], 'b-', label='Trajectory (XY)')
    plt.scatter(odom_data_np[0, 0], odom_data_np[0, 1], color='green', marker='o', s=100, label='Start')
    plt.scatter(odom_data_np[-1, 0], odom_data_np[-1, 1], color='red', marker='X', s=100, label='End')
    plt.title('Relative Odometry Trajectory on Static 3D Lidar (XY Plane)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.tight_layout()
    
    plt_file = os.path.join(ws_folder, "hesai_odom_trajectory_xy.png")
    plt.savefig(plt_file, dpi=300)
    plt.show()
    plt.close()
    print(f"[✔] Figura salva em '{plt_file}'")


    odom_file = "hesai_odom_data.npy"
    output_file = os.path.join(ws_folder, odom_file)
    np.save(output_file, odom_data_np)
    print(f"[✔] Odometria guardada em '{output_file}' com {len(odom_data)} amostras.")
    
    #smooth the trajectory using Savitzky-Golay filter
    from scipy.signal import savgol_filter
    #remove z component for smoothing
    odom_data_np = odom_data_np[:, :2]  # Keep only x and y for smoothing
    if len(odom_data_np) > 5:  # Ensure enough points for smoothing
        smoothed_odom = savgol_filter(odom_data_np, window_length=10, polyorder=2, axis=0)
        smoothed_file = os.path.join(ws_folder, "hesai_odom_data_smoothed.npy")
        np.save(smoothed_file, smoothed_odom)
        print(f"[✔] Odometria suavizada guardada em '{smoothed_file}' com {len(smoothed_odom)} amostras.")
        
        plt.figure(figsize=(8, 6))
        plt.plot(odom_data_np[:, 0], odom_data_np[:, 1], 'b-', label='Original Trajectory (XY)')
        plt.plot(smoothed_odom[:, 0], smoothed_odom[:, 1], 'r-', label='Smoothed Trajectory (XY)')
        plt.scatter(odom_data_np[0, 0], odom_data_np[0, 1], color='green', marker='o', s=100, label='Start')
        plt.scatter(odom_data_np[-1, 0], odom_data_np[-1, 1], color='red', marker='X', s=100, label='End')
        plt.title('Relative Odometry Trajectory on Static 3D Lidar (XY Plane) - Smoothed')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.tight_layout()
        smoothed_plot_file = os.path.join(ws_folder, "hesai_odom_trajectory_xy_smoothed.png")
        plt.savefig(smoothed_plot_file, dpi=300)
        plt.show()
        print(f"[✔] Figura suavizada salva em '{smoothed_plot_file}'")
    

if __name__ == "__main__":
    import sys
    import os
    if len(sys.argv) < 2:
        print("Uso: python lidar_odom.py /caminho/para/rosbag ")
        bag_path = "/home/olivas/camera_ws/bags/rosbag2_2025_09_16-11_50_59"
    else:
        bag_path = sys.argv[1]
        
    # Create output directory for odometry results
    bag_folder = os.path.dirname(os.path.normpath(bag_path))
    print(f"[INFO] Bag folder: {bag_folder}")
    
    hesai_odom_dir = os.path.join(bag_folder, "hesai_odom")
    os.makedirs(hesai_odom_dir, exist_ok=True)
    ws_folder = hesai_odom_dir  # Use this as output folder for plots and npy
    print(f"[INFO] Criando diretório de saída: {hesai_odom_dir}")
    

    ground_plane_file = "/home/olivas/camera_ws/src/Tracking_Tool_3D_Lidar/ground_plane_background_full.npy"
    #ground_plane_file = "/home/olivas/camera_ws/src/Tracking_Tool_3D_Lidar/ground_plane_enhanced.npy"

    if os.path.exists(ground_plane_file):
        ground_plane = np.load(ground_plane_file)
        print(f"[INFO] Loaded ground plane from '{ground_plane_file}': {ground_plane}")
    else:
        print(f"[ERROR] Ground plane file not found: {ground_plane_file}")
        exit(1)

    
    # Get the first folder in the bag_path (relative to current directory)
    
    # Search for a file with 'ground_plane' in its name in the bag directory
    
    """ ground_plane_file = None
    print(f"[INFO] Procurando arquivo de ground plane em '{bag_folder}'...")
    for fname in os.listdir(bag_folder):
        if "ground_plane" in fname:
            ground_plane_file = os.path.join(bag_folder, fname)
            break
    print(f"[INFO] Extraindo odometria de '{bag_path}' para '{output_file}'...")
    if ground_plane_file:
        print(f"[INFO] Arquivo de ground plane encontrado: '{ground_plane_file}'")
        # Load the ground plane file to ensure it exists
        try:
            ground_plane = np.load(ground_plane_file)
            print(f"[INFO] Ground plane carregado com sucesso: {ground_plane.shape}")
        except Exception as e:
            print(f"[ERROR] Falha ao carregar o arquivo de ground plane: {e}")
            exit(1)
    else:
        print("[INFO] Nenhum arquivo de ground plane encontrado.")
        exit(1) """
    
    
    
    
    rclpy.init()
    extract_odom_from_rosbag(bag_path,ground_plane=ground_plane,ws_folder=ws_folder)
    rclpy.shutdown()
