import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import pclpy
from pclpy import pcl
from pathlib import Path
from typing import Tuple


def adaptive_voxel_size(
    points: np.ndarray,
    num_neighbors: int = 3,
    grid_size: int = 10,
    target_size_mb: int = 60,
) -> float:
    """Determine the voxel size based on the average distance to the nearest neighbors and target file size."""
    current_size_mb = len(points) * 12 / (1024 * 1024)
    if current_size_mb <= target_size_mb:
        return 0  # No downsampling if already below target size

    # Calculate the bounding box of the point cloud
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    grid_step = (max_bounds - min_bounds) / grid_size

    # Select a random grid cell
    grid_indices = np.random.randint(0, grid_size, size=3)
    cell_min = min_bounds + grid_indices * grid_step
    cell_max = cell_min + grid_step

    # Filter points within the selected grid cell
    mask = np.all((points >= cell_min) & (points < cell_max), axis=1)
    region_points = points[mask]

    if len(region_points) < num_neighbors:
        return 0  # No downsampling if not enough points in the region

    # Calculate nearest neighbors within the region
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm="auto").fit(
        region_points
    )
    distances, _ = nbrs.kneighbors(region_points)

    # Exclude the first column (distance to itself)
    avg_distance = np.mean(distances[:, 1:])

    # Calculate target point count for 20 MB
    target_point_count = target_size_mb * 1024 * 1024 // 24

    # Adjust voxel size based on average distance and target point count
    reduction_factor = len(points) / target_point_count
    voxel_size = avg_distance * (reduction_factor ** (1 / 3)) * 10
    return voxel_size


def downsample_with_open3d(points: np.ndarray, file_size: int) -> np.ndarray:
    """Downsample the pointcloud using Open3D's voxel grid filtering."""
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Determine adaptive voxel size
    voxel_size = adaptive_voxel_size(points)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # Convert back to numpy array
    downsampled_points = np.asarray(pcd.points)
    return downsampled_points


def load_and_downsample_file(
    content: bytes, file_size: int, file_format: str, chunk_size: int = 50000
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and downsample pointcloud data from file content in chunks for all supported formats, including color for PLY files."""
    try:
        points = []
        colors = []
        if file_format == ".ply":
            # Write the content to a temporary file
            temp_file_path = "temp.ply"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(content)

            # Use Open3D to read the PLY file in chunks
            pcd = o3d.io.read_point_cloud(temp_file_path)
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)

            # Downsample using Open3D
            points = downsample_with_open3d(points, file_size)

            # Remove the temporary file
            Path(temp_file_path).unlink()

        # Concatenate all downsampled chunks
        points = np.concatenate(points, axis=0)
        if colors.size > 0:
            colors = np.concatenate(colors, axis=0)
        return points, colors
    except Exception as e:
        print(f"Error loading {file_format} file: {e}")
        return None, None


def generate_colors(points: np.ndarray, color_mode) -> np.ndarray:
    """Generate colors for points based on current color mode."""
    if points is None:
        return None

    if color_mode == "Height":
        # Normalize z values to 0-1 range
        z_normalized = (points[:, 2] - points[:, 2].min()) / (
            points[:, 2].max() - points[:, 2].min()
        )
        # Create a green-to-brown color gradient
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 1] = (z_normalized * 255).astype(np.uint8)  # Green channel
        colors[:, 0] = ((1 - z_normalized) * 100).astype(np.uint8)  # Red channel

    return colors


def save_downsampled_pointcloud(points: np.ndarray, original_file_path: str) -> str:
    """Save the downsampled pointcloud to a file in the same directory as the original file."""
    downsampled_file_path = original_file_path + "_downsampled.ply"
    pcd = o3d.geometry.PointCloud()

    # Reshape points to have three columns
    if points.size % 3 != 0:
        raise ValueError(
            "The total number of elements in points is not divisible by 3."
        )
    points = points.reshape((-1, 3))
    print(f"Points array shape after reshaping: {points.shape}")

    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(downsampled_file_path, pcd)
    return downsampled_file_path


def load_downsampled_pointcloud(original_file_path: str) -> np.ndarray:
    """Load the downsampled pointcloud if it exists."""
    downsampled_file_path = original_file_path + "_downsampled.ply"
    if Path(downsampled_file_path).exists():
        pcd = o3d.io.read_point_cloud(downsampled_file_path)
        return np.asarray(pcd.points)
    return None


def radius_outlier_removal(
    points: np.ndarray,
    colors: np.ndarray,
    avg_distance: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:

    from ipdb import set_trace

    set_trace()

    # Convert points to PCL format
    cloud = pcl.PointCloud.PointXYZ(points)

    # Apply Radius Outlier Removal
    ror = pcl.filters.RadiusOutlierRemoval.PointXYZ()
    ror.setInputCloud(cloud)
    ror.setRadiusSearch(avg_distance * 2)
    ror.setMinNeighborsInRadius(6)
    filtered_cloud = pcl.PointCloud.PointXYZ()
    ror.filter(filtered_cloud)

    filtered_points = np.asarray(filtered_cloud.xyz)

    if colors is not None:
        # Convert rows to tuples for matching
        point_set = set(map(tuple, filtered_points))
        mask = np.array([tuple(p) in point_set for p in points])

        from ipdb import set_trace

        set_trace()

        filtered_colors = colors[mask]
        return filtered_points, filtered_colors

    return filtered_points, None
