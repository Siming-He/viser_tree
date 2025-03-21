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


class ForestVisualizer:
    def __init__(self):
        self.server = viser.ViserServer()
        self.setup_gui()
        self.current_points = None
        self.current_colors = None

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

    def adaptive_voxel_size(
        self,
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

    def downsample_with_open3d(self, points: np.ndarray, file_size: int) -> np.ndarray:
        """Downsample the pointcloud using Open3D's voxel grid filtering."""
        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Determine adaptive voxel size
        voxel_size = self.adaptive_voxel_size(points)
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size)

        # Convert back to numpy array
        downsampled_points = np.asarray(pcd.points)
        return downsampled_points

    def load_and_downsample_file(
        self, content: bytes, file_size: int, file_format: str, chunk_size: int = 50000
    ) -> np.ndarray:
        """Load and downsample pointcloud data from file content in chunks for all supported formats."""
        try:
            points = []

            if file_format in [".xyz", ".txt"]:
                # Convert bytes to string and split into lines
                text = content.decode("utf-8")
                lines = text.strip().split("\n")

                # Process in chunks
                for i in range(0, len(lines), chunk_size):
                    chunk_lines = lines[i : i + chunk_size]
                    chunk_points = []
                    for line in chunk_lines:
                        values = line.strip().split()
                        if len(values) >= 3:  # Ensure we have at least x, y, z
                            x, y, z = map(float, values[:3])
                            chunk_points.append([x, y, z])

                    # Convert to numpy array
                    chunk_points = np.array(chunk_points, dtype=np.float32)

                    # Downsample using Open3D
                    chunk_points = self.downsample_with_open3d(chunk_points, file_size)

                    # Append downsampled points
                    points.append(chunk_points)

            elif file_format == ".pcd":
                # Check if the file is binary or ASCII
                if b"DATA binary" in content:
                    # Binary PCD file
                    header_end = content.find(b"DATA binary") + len(b"DATA binary")
                    header = content[:header_end].decode("utf-8")
                    data = content[header_end:].strip()

                    # Parse header
                    lines = header.split("\n")
                    header_info = {}
                    for line in lines:
                        if line.startswith("FIELDS"):
                            header_info["fields"] = line.split()[1:]
                        elif line.startswith("SIZE"):
                            header_info["size"] = list(map(int, line.split()[1:]))
                        elif line.startswith("TYPE"):
                            header_info["type"] = line.split()[1:]
                        elif line.startswith("COUNT"):
                            header_info["count"] = list(map(int, line.split()[1:]))
                        elif line.startswith("WIDTH"):
                            header_info["width"] = int(line.split()[1])
                        elif line.startswith("HEIGHT"):
                            header_info["height"] = int(line.split()[1])
                        elif line.startswith("POINTS"):
                            header_info["points"] = int(line.split()[1])

                    # Read binary data in chunks
                    point_count = header_info["points"]
                    point_size = sum(header_info["size"])
                    for i in range(0, point_count, chunk_size):
                        chunk_data = data[
                            i * point_size : (i + chunk_size) * point_size
                        ]
                        chunk_points = np.frombuffer(
                            chunk_data,
                            dtype=np.float32,
                            count=min(chunk_size, point_count - i) * 3,
                        )
                        chunk_points = chunk_points.reshape((-1, 3))

                        # Downsample using Open3D
                        chunk_points = self.downsample_with_open3d(
                            chunk_points, file_size
                        )

                        # Append downsampled points
                        points.append(chunk_points)

                else:
                    # ASCII PCD file
                    text = content.decode("utf-8")
                    lines = text.strip().split("\n")

                    # Parse header
                    header = {}
                    data_idx = 0
                    for i, line in enumerate(lines):
                        if line.startswith("#"):
                            continue

                        if line.startswith("VERSION"):
                            header["version"] = line.split()[1]
                        elif line.startswith("FIELDS"):
                            header["fields"] = line.split()[1:]
                        elif line.startswith("SIZE"):
                            header["size"] = list(map(int, line.split()[1:]))
                        elif line.startswith("TYPE"):
                            header["type"] = line.split()[1:]
                        elif line.startswith("COUNT"):
                            header["count"] = list(map(int, line.split()[1:]))
                        elif line.startswith("WIDTH"):
                            header["width"] = int(line.split()[1])
                        elif line.startswith("HEIGHT"):
                            header["height"] = int(line.split()[1])
                        elif line.startswith("POINTS"):
                            header["points"] = int(line.split()[1])
                        elif line.startswith("DATA"):
                            header["data_format"] = line.split()[1]
                            data_idx = i + 1
                            break

                    # Process in chunks
                    for i in range(data_idx, len(lines), chunk_size):
                        chunk_lines = lines[i : i + chunk_size]
                        chunk_points = []
                        for line in chunk_lines:
                            if line.strip():
                                values = line.strip().split()
                                if len(values) >= 3:
                                    x, y, z = map(float, values[:3])
                                    chunk_points.append([x, y, z])

                        chunk_points = np.array(chunk_points, dtype=np.float32)

                        # Downsample using Open3D
                        chunk_points = self.downsample_with_open3d(
                            chunk_points, file_size
                        )

                        # Append downsampled points
                        points.append(chunk_points)

            elif file_format == ".ply":
                # Write the content to a temporary file
                temp_file_path = "temp.ply"
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(content)

                # Use Open3D to read the PLY file in chunks
                pcd = o3d.io.read_point_cloud(temp_file_path)
                points = np.asarray(pcd.points)
                self.current_colors = (
                    np.asarray(pcd.colors) if pcd.has_colors() else None
                )

                # Downsample using Open3D
                points = self.downsample_with_open3d(points, file_size)

                # Remove the temporary file
                Path(temp_file_path).unlink()

            # Concatenate all downsampled chunks
            points = np.concatenate(points, axis=0)
            return points
        except Exception as e:
            print(f"Error loading {file_format} file: {e}")
            return None

    def generate_colors(self, points: np.ndarray) -> np.ndarray:
        """Generate colors for points based on current color mode."""
        if points is None:
            return None

        if self.color_mode.value == "Height":
            # Normalize z values to 0-1 range
            z_normalized = (points[:, 2] - points[:, 2].min()) / (
                points[:, 2].max() - points[:, 2].min()
            )
            # Create a green-to-brown color gradient
            colors = np.zeros((len(points), 3), dtype=np.uint8)
            colors[:, 1] = (z_normalized * 255).astype(np.uint8)  # Green channel
            colors[:, 0] = ((1 - z_normalized) * 100).astype(np.uint8)  # Red channel
        else:
            colors = np.tile(self.color_picker.value, (len(points), 1))

        return colors

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
            colors = self.generate_colors(self.current_points)

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

    async def load_pointcloud_in_chunks(
        self, points: np.ndarray, chunk_size: int = 50000
    ) -> None:
        """Load the pointcloud in chunks to improve performance."""
        num_points = len(points)
        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            chunk = points[start:end]
            self.current_points = chunk
            self.update_visualization()
            await asyncio.sleep(0.05)  # Yield control to keep UI responsive

    def filter_and_process_pointcloud(self, points: np.ndarray) -> dict:
        """Filter and process the point cloud to identify ground points and tree structures."""
        # Step 1: Filter ground points using morphological filtering
        ground_points, non_ground_points = self.floor_remove(points)

        # Step 2: Remove outliers
        filtered_points = self.radius_outlier_removal(non_ground_points)

        # Step 3: Cluster points using Euclidean distance
        cluster_list = self.euclidean_cluster_extract(filtered_points)

        print(f"Cluster list length: {len(cluster_list)}")

        # Step 4: Further refine clusters using region growing
        tree_clusters = []
        for cluster in cluster_list:
            rg_clusters = self.region_growing(cluster)
            for rg_cluster in rg_clusters:
                tree_clusters.append(rg_cluster)

        # Step 5: Store data in the required format
        tree_data = {
            "ground_points": ground_points,
            "trees": [
                (cluster, []) for cluster in tree_clusters
            ],  # Placeholder for foliage
        }

        return tree_data

    def setup_tree_list(self, tree_data: dict):
        """Setup the GUI to list and visualize each tree."""
        # Debugging output to check the contents of tree_data
        print(f"Tree data: {tree_data}")

        with self.server.gui.add_folder("Tree List"):
            for i, (trunk, foliage) in enumerate(tree_data["trees"]):
                self.server.gui.add_text(
                    f"Tree {i+1}",
                    initial_value=f"Trunk Points: {len(trunk)}, Foliage Points: {len(foliage)}",
                )

        # Visualize ground
        self.server.scene.add_point_cloud(
            "/ground_points",
            points=tree_data["ground_points"],
            colors=np.array(
                [[150, 75, 0]] * len(tree_data["ground_points"])
            ),  # Brown for ground
            position=(0, 0, 0),
            point_size=self.point_size.value,
            point_shape="circle",
        )

        # Visualize all trees
        for i, (trunk, foliage) in enumerate(tree_data["trees"]):
            self.server.scene.add_point_cloud(
                f"/tree_{i+1}_trunk",
                points=trunk,
                colors=np.array([[255, 0, 0]] * len(trunk)),  # Red for trunk
                position=(0, 0, 0),
                point_size=self.point_size.value,
                point_shape="circle",
            )
            self.server.scene.add_point_cloud(
                f"/tree_{i+1}_foliage",
                points=foliage,
                colors=np.array([[0, 255, 0]] * len(foliage)),  # Green for foliage
                position=(0, 0, 0),
                point_size=self.point_size.value,
                point_shape="circle",
            )

        # Ensure visualization is updated
        # self.update_visualization()

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

                # Load points based on file format
                points = None
                if filename.endswith(".xyz") or filename.endswith(".txt"):
                    points = self.load_and_downsample_file(content, file_size, ".xyz")
                elif filename.endswith(".pcd"):
                    points = self.load_and_downsample_file(content, file_size, ".pcd")
                elif filename.endswith(".ply"):
                    points = self.load_and_downsample_file(content, file_size, ".ply")
                else:
                    print(
                        "Unsupported file format. Please upload .xyz, .txt, .pcd, or .ply files"
                    )
                    return

                if points is not None and len(points) > 0:
                    # Reshape points to ensure it's a 2D array
                    points = points.reshape(-1, 3)

                    # Filter and process the point cloud

                    print("Filtering and processing point cloud...")
                    tree_data = self.filter_and_process_pointcloud(points)

                    # Setup tree list in the GUI
                    print("Setting up tree list...")
                    self.setup_tree_list(tree_data)

                    print("Setting up tree list... done")

                    # Calculate average nearest neighbor distance
                    nbrs = NearestNeighbors(n_neighbors=2).fit(points)
                    distances, _ = nbrs.kneighbors(points)
                    avg_distance = np.mean(distances[:, 1])
                    initial_point_size = avg_distance * 0.1
                    print(f"Initial point size: {initial_point_size}")
                    self.point_size.value = initial_point_size
                    self.point_size.min = max(0, initial_point_size - 0.1)
                    self.point_size.max = initial_point_size + 0.1
                    self.current_points = points
                    self.update_visualization()
                    self.point_count_text.value = f"Loaded {len(points)} points"
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

    def floor_remove(self, points: np.ndarray) -> tuple:
        # Convert points to PCL format
        cloud = pcl.PointCloud.PointXYZ(points)

        # Apply Progressive Morphological Filter
        pmf = pcl.segmentation.ApproximateProgressiveMorphologicalFilter.PointXYZ()
        pmf.setInputCloud(cloud)
        pmf.setMaxWindowSize(20)
        pmf.setSlope(1.0)
        pmf.setInitialDistance(0.5)
        pmf.setMaxDistance(3.0)
        ground_indices = pcl.vectors.Int()
        pmf.extract(ground_indices)

        # Extract ground and non-ground points
        extract = pcl.filters.ExtractIndices.PointXYZ()
        extract.setInputCloud(cloud)
        extract.setIndices(ground_indices)
        ground = pcl.PointCloud.PointXYZ()
        extract.filter(ground)
        extract.setNegative(True)
        non_ground = pcl.PointCloud.PointXYZ()
        extract.filter(non_ground)

        return np.asarray(ground.xyz), np.asarray(non_ground.xyz)

    def radius_outlier_removal(self, points: np.ndarray) -> np.ndarray:
        # Convert points to PCL format
        cloud = pcl.PointCloud.PointXYZ(points)

        # Apply Radius Outlier Removal
        ror = pcl.filters.RadiusOutlierRemoval.PointXYZ()
        ror.setInputCloud(cloud)
        ror.setRadiusSearch(0.4)
        ror.setMinNeighborsInRadius(6)
        filtered_cloud = pcl.PointCloud.PointXYZ()
        ror.filter(filtered_cloud)

        return np.asarray(filtered_cloud.xyz)

    def euclidean_cluster_extract(self, points: np.ndarray) -> list:
        # Convert points to PCL format
        cloud = pcl.PointCloud.PointXYZI(points)

        # Estimate normals
        ne = pcl.features.NormalEstimationOMP.PointXYZI_Normal()
        tree = pcl.search.KdTree.PointXYZI()
        ne.setInputCloud(cloud)
        ne.setSearchMethod(tree)
        ne.setRadiusSearch(0.035)
        normals = pcl.PointCloud.Normal()
        ne.compute(normals)

        # Create the segmentation object for cylinder segmentation
        seg = pcl.segmentation.SACSegmentationFromNormals.PointXYZI_Normal()
        seg.setOptimizeCoefficients(True)
        seg.setModelType(pcl.sample_consensus.SACMODEL_CYLINDER)
        seg.setMethodType(pcl.sample_consensus.SAC_RANSAC)
        seg.setMaxIterations(1000)
        seg.setDistanceThreshold(0.025)
        seg.setRadiusLimits(0.025, 0.35)
        seg.setInputCloud(cloud)
        seg.setInputNormals(normals)

        # Obtain the cylinder inliers and coefficients
        inliers = pcl.PointIndices()
        coefficients = pcl.ModelCoefficients()
        seg.segment(inliers, coefficients)

        # Extract the cylindrical component
        extract = pcl.filters.ExtractIndices.PointXYZI()
        extract.setInputCloud(cloud)
        extract.setIndices(inliers)
        extract.setNegative(False)
        stem = pcl.PointCloud.PointXYZI()
        extract.filter(stem)

        # Convert back to numpy array
        return [np.asarray(stem.xyz)]

    def region_growing(self, points: np.ndarray) -> list:
        # Example logic for region growing
        clusters = []  # Replace with actual region growing logic
        # Placeholder: Assume clusters is a list of lists
        for cluster in clusters:
            clusters.append(np.array(cluster))  # Ensure each cluster is a NumPy array
        return clusters


if __name__ == "__main__":
    visualizer = ForestVisualizer()
    visualizer.run()
