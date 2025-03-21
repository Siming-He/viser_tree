import open3d as o3d
import numpy as np
import os
import argparse

def calculate_voxel_size(points, target_size_mb=60):
    """Calculate the voxel size to downsample the point cloud to approximately 60 MB."""
    current_size_mb = len(points) * 12 / (1024 * 1024)  # Approximate size of a point in bytes (3 floats)
    if current_size_mb <= target_size_mb:
        return 0  # No downsampling needed

    # Calculate target point count for 60 MB
    target_point_count = target_size_mb * 1024 * 1024 // 12
    reduction_factor = len(points) / target_point_count

    # Estimate voxel size based on reduction factor
    voxel_size = np.cbrt(reduction_factor) * 10  # Adjust the multiplier as needed
    return voxel_size

def downsample_pointcloud(input_file, target_size_mb=60):
    """Downsample a point cloud to approximately 60 MB."""
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)

    # Before converting to Vector3dVector
    print(f"Points array shape: {points.shape}")
    print(f"Points array dtype: {points.dtype}")

    # Calculate voxel size
    voxel_size = calculate_voxel_size(points, target_size_mb)
    if voxel_size > 0:
        # Downsample the point cloud
        pcd = pcd.voxel_down_sample(voxel_size)

    # Convert numpy array to Open3D point cloud
    pcd.points = o3d.utility.Vector3dVector(points)

    # Determine output file path
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_downsampled.ply"

    # Save the downsampled point cloud
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Downsampled point cloud saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample a point cloud to approximately 60 MB.")
    parser.add_argument("input_file", type=str, help="Path to the input point cloud file.")
    args = parser.parse_args()

    downsample_pointcloud(args.input_file)