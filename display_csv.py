#!/usr/bin/env python3
"""
display_csv_path.py
Display multiple car paths from CSV files in the same visualization windows
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

def load_csv_path(csv_file):
    """Load the car path from CSV file"""
    df = pd.read_csv(csv_file)
    positions = df[['x', 'y', 'z']].values
    orientations = df[['qx', 'qy', 'qz', 'qw']].values
    timestamps = df['timestamp'].values
    return positions, orientations, timestamps

def plot_3d_trajectory(positions_list, labels, output_file=None):
    """Plot multiple 3D trajectories in the same figure"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y']  # Colors for different trajectories
    for i, (positions, label) in enumerate(zip(positions_list, labels)):
        color = colors[i % len(colors)]
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=color, label=label, linewidth=2)
    
    ax.set_title('Car Trajectories Comparison (3D)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()

def plot_3d_trajectory_equal_axes(positions_list, labels, output_file=None):
    """Plot multiple 3D trajectories with equal axes"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    for i, (positions, label) in enumerate(zip(positions_list, labels)):
        color = colors[i % len(colors)]
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=color, label=label, linewidth=2)
    
    ax.set_title('Car Trajectories Comparison (3D, axis equal)')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()

    # Set equal aspect ratio for 3D axes
    def set_axes_equal(ax):
        all_positions = np.vstack(positions_list)
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
        
        max_range = max([x_max - x_min, y_max - y_min, z_max - z_min])
        x_middle = (x_max + x_min) / 2
        y_middle = (y_max + y_min) / 2
        z_middle = (z_max + z_min) / 2
        
        ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
        ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
        ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    set_axes_equal(ax)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()

def plot_xy_trajectory(positions_list, labels, output_file=None):
    """Plot multiple XY trajectories in the same figure"""
    plt.figure(figsize=(10, 8))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (positions, label) in enumerate(zip(positions_list, labels)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(positions[:, 0], positions[:, 1], 
                color=color, label=label, linewidth=2)
        plt.scatter(positions[0, 0], positions[0, 1], 
                   color=color, marker=marker, s=100, label=f'{label} Start')
        plt.scatter(positions[-1, 0], positions[-1, 1], 
                   color=color, marker='X', s=150, label=f'{label} End')

    plt.title('Car Trajectories Comparison (XY Plane)')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()

def calculate_velocity(positions, timestamps):
    """Calculate velocity profile"""
    if len(positions) > 1:
        dt = np.diff(timestamps)
        
        # Handle zero time differences to avoid division by zero
        dt[dt == 0] = np.finfo(float).eps
        
        dx = np.diff(positions[:, 0])
        dy = np.diff(positions[:, 1])
        dz = np.diff(positions[:, 2])
        
        velocity = np.sqrt(dx**2 + dy**2 + dz**2) / dt
        return velocity
    return None

def plot_velocity_profiles(positions_list, timestamps_list, labels, output_file=None):
    """Plot multiple velocity profiles in the same figure"""
    plt.figure(figsize=(12, 6))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y']
    
    for i, (positions, timestamps, label) in enumerate(zip(positions_list, timestamps_list, labels)):
        velocity = calculate_velocity(positions, timestamps)
        if velocity is not None:
            color = colors[i % len(colors)]
            plt.plot(velocity, color=color, label=label, linewidth=2)
    
    plt.title('Velocity Profiles Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()

def main(csv_files, output_dir=None):
    """Main function to display multiple CSV paths"""
    positions_list = []
    orientations_list = []
    timestamps_list = []
    labels = []
    
    # Load data from all CSV files
    for csv_file in csv_files:
        try:
            positions, orientations, timestamps = load_csv_path(csv_file)
            positions_list.append(positions)
            orientations_list.append(orientations)
            timestamps_list.append(timestamps)
            
            # Create label from filename
            filename = os.path.basename(csv_file)
            label = os.path.splitext(filename)[0]
            labels.append(label)
            
            print(f"Loaded {len(positions)} poses from {csv_file}")
            print(f"  X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
            print(f"  Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")
            print(f"  Z range: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}]")
            print()
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not positions_list:
        print("No valid CSV files loaded!")
        return
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Plot 3D trajectories
    plot_3d_trajectory(positions_list, labels, 
                      output_file=os.path.join(output_dir, "comparison_trajectory_3d.png") if output_dir else None)
    
    # Plot 3D trajectories with equal axes
    plot_3d_trajectory_equal_axes(positions_list, labels, 
                                 output_file=os.path.join(output_dir, "comparison_trajectory_3d_equal.png") if output_dir else None)
    
    # Plot XY trajectories
    plot_xy_trajectory(positions_list, labels, 
                      output_file=os.path.join(output_dir, "comparison_trajectory_xy.png") if output_dir else None)
    
    # Plot velocity profiles
    plot_velocity_profiles(positions_list, timestamps_list, labels, 
                          output_file=os.path.join(output_dir, "comparison_velocity_profiles.png") if output_dir else None)
    
    # Save processed data as numpy arrays
    if output_dir:
        for i, (positions, label) in enumerate(zip(positions_list, labels)):
            np.save(os.path.join(output_dir, f"{label}_trajectory_data.npy"), positions)
        print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) < 2:
        print("Usage: python display_csv_path.py <path_to_csv_file1> [path_to_csv_file2 ...] [output_directory]")
        print("Example: python display_csv_path.py car_trajectory.csv amcl_path.csv ./comparison_output")
        print("Or use wildcards: python display_csv_path.py *.csv ./output")
        sys.exit(1)
    
    # Separate CSV files from output directory (last argument if it's a directory)
    args = sys.argv[1:]
    csv_files = []
    output_dir = None
    
    for arg in args:
        if os.path.isdir(arg):
            output_dir = arg
        elif '*' in arg or '?' in arg:
            # Handle wildcards
            csv_files.extend(glob.glob(arg))
        elif arg.endswith('.csv'):
            csv_files.append(arg)
    
    # If no CSV files found, check if the last argument might be output dir
    if not csv_files and len(args) > 1:
        csv_files = [arg for arg in args if arg.endswith('.csv')]
        if len(csv_files) < len(args):
            output_dir = args[-1] if os.path.isdir(args[-1]) else None
    
    # Remove output directory from CSV files list if it was mistakenly added
    if output_dir in csv_files:
        csv_files.remove(output_dir)
    
    if not csv_files:
        print("Error: No CSV files found!")
        print("Usage: python display_csv_path.py <path_to_csv_file1> [path_to_csv_file2 ...] [output_directory]")
        sys.exit(1)
    
    # Check if all CSV files exist
    missing_files = [f for f in csv_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: The following files were not found:")
        for f in missing_files:
            print(f"  {f}")
        sys.exit(1)
    
    print(f"Comparing {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f}")
    print()
    
    main(csv_files, output_dir)