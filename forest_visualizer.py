"""Forest Pointcloud Visualizer

A simple visualizer for forest pointcloud data with GUI controls for visualization parameters.
"""

import numpy as np
import viser
from pathlib import Path
import time
import struct
import open3d as o3d
import asyncio
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import pclpy
from pclpy import pcl
import pointcloud_processing as pc_proc
from typing import Tuple


class ForestVisualizer:
    def __init__(self):
        self.server = viser.ViserServer()
        self.setup_gui()
        self.current_points = None
        self.current_colors = None
        self.avg_distance = 0.1  # Default value
        self.color_mode = self.server.gui.add_dropdown(
            "Color Mode", options=["Height", "Original"], initial_value="Original"
        )

    def setup_gui(self):
        # File upload section
        with self.server.gui.add_folder("Data Loading"):
            self.upload_button = self.server.gui.add_upload_button(
                "Load Pointcloud", icon=viser.Icon.UPLOAD
            )
            self.point_count_text = self.server.gui.add_text(
                "Point Count", initial_value="No points loaded", disabled=True
            )

        # Visualization controls
        with self.server.gui.add_folder("Visualization Controls"):
            self.point_size = self.server.gui.add_slider(
                "Point Size",
                min=0.1,
                max=5.0,
                step=0.001,
                initial_value=1.0,
            )

    def update_visualization(self):
        """Update the pointcloud visualization with current settings."""
        if self.current_points is None:
            return

        # Debugging output to check the shape of self.current_points
        # print(f"Current points shape before generating colors: {self.current_points.shape}")

        # Use original colors if available
        if self.current_colors is not None:
            colors = self.current_colors
        else:
            self.color_mode.value = "Height"
            colors = pc_proc.generate_colors(self.current_points, self.color_mode.value)

        print(f"Current points shape: {self.current_points.shape}")
        print(f"Colors shape: {colors.shape}")

        # Update the pointcloud
        self.server.scene.add_point_cloud(
            "/forest_points",
            points=self.current_points,
            colors=colors,
            position=(0, 0, 0),
            point_size=self.point_size.value,
            point_shape="circle",
        )

    def save_downsampled_pointcloud(
        self, points: np.ndarray, original_file_path: str
    ) -> str:
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

    def load_downsampled_pointcloud(self, original_file_path: str) -> np.ndarray:
        """Load the downsampled pointcloud if it exists."""
        downsampled_file_path = original_file_path + "_downsampled.ply"
        if Path(downsampled_file_path).exists():
            pcd = o3d.io.read_point_cloud(downsampled_file_path)
            return np.asarray(pcd.points)
        return None

    def run(self):
        @self.upload_button.on_upload
        def _(event) -> None:
            """Handle file upload events."""
            try:
                # Print event object for debugging
                print(f"Upload event received: {event}")
                print(f"Event attributes: {dir(event)}")

                # Access the file data from the target attribute
                file_data = self.upload_button.value
                if not file_data:
                    print("No file content received")
                    return

                filename = file_data.name
                content = file_data.content
                file_size = len(content)

                print(f"Processing file: {filename}")

                # Load points and colors based on file format
                points, colors = None, None
                if filename.endswith(".ply"):
                    points, colors = pc_proc.load_and_downsample_file(
                        content, file_size, ".ply"
                    )
                    print(f"Points shape: {points.shape}")
                    print(f"Colors shape: {colors.shape}")
                else:
                    print("Unsupported file format. Please upload .ply files")
                    return

                if points is not None and len(points) > 0:
                    # Reshape points to ensure it's a 2D array
                    points = points.reshape(-1, 3)

                    if self.avg_distance == 0.1:
                        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
                        distances, _ = nbrs.kneighbors(points)
                        self.avg_distance = np.mean(distances[:, 1])

                    # Filter and process the point cloud
                    if colors is not None and colors.size > 0:

                        from ipdb import set_trace

                        set_trace()

                        points, self.current_colors = pc_proc.radius_outlier_removal(
                            points, colors.reshape(-1, 3), self.avg_distance
                        )
                        print(f"Current colors shape: {self.current_colors.shape}")
                    else:
                        points, self.current_colors = pc_proc.radius_outlier_removal(
                            points, None, self.avg_distance
                        )
                    # Calculate average nearest neighbor distance if not set

                    # Calculate initial point size
                    initial_point_size = self.avg_distance * 0.2
                    print(f"Initial point size: {initial_point_size}")
                    self.point_size.value = initial_point_size
                    self.point_size.min = max(0, initial_point_size - 0.1)
                    self.point_size.max = initial_point_size + 0.1
                    self.current_points = points
                    self.update_visualization()
                    self.point_count_text.value = f"Loaded {len(points)} points"
                    print(f"Number of points: {len(points)}")
                    if colors is not None:
                        print(f"Number of colors: {len(colors)}")
                    else:
                        print("No colors loaded.")
                else:
                    print("Failed to load points from file")
            except Exception as e:
                print(f"Error processing file: {e}")
                import traceback

                traceback.print_exc()

        # Set up callbacks for GUI changes
        self.point_size.on_update(lambda _: self.update_visualization())

        print(
            "Forest Pointcloud Visualizer is running. Please open a web browser and navigate to:"
        )
        print("http://localhost:8080")

        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down the visualizer...")


if __name__ == "__main__":
    visualizer = ForestVisualizer()
    visualizer.run()
