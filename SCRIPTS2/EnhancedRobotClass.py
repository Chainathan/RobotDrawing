# Improved Robot Portrait Drawing with Tkinter/Matplotlib fixes
# ==============================

import os
import cv2
import numpy as np
import matplotlib
# Set non-interactive backend to avoid Tkinter errors
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance
from skimage.morphology import skeletonize, remove_small_objects
from scipy.interpolate import interp1d
from collections import deque
import time
from tqdm import tqdm

# Improved RobotClass with enhanced drawing capabilities
class EnhancedRobotClass:
    def __init__(self):
        '''
        The units are in millimeters. It will be converted to meters before feeding to the robot.
        '''
        # Set canvas dimensions (a 0.25m square centered at (0.375, 0))
        side = 0.25
        self.canvas_size = (side, side)
        self.canvas_center = (0.375, 0)
        self.canvas_bounds_x = (self.canvas_center[0] - side/2, self.canvas_center[0] + side/2)
        self.canvas_bounds_y = (self.canvas_center[1] - side/2, self.canvas_center[1] + side/2)
        self.origin = (self.canvas_bounds_x[0], self.canvas_bounds_y[0])

        # Improved height parameters for more precise drawing
        self.pen_up_height = 0.125     # Higher pen-up to avoid dragging
        self.pen_down_height = 0.095   # Slightly adjusted pen-down for better contact
        self.pen_height_diff = self.pen_up_height - self.pen_down_height

        # Initialize the robot interface
        try:
            from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
            from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
            self.bot = InterbotixManipulatorXS(
                robot_model='wx250s',
                group_name='arm',
                gripper_name='gripper',
            )
            robot_startup()
            self._simulation_mode = False
            print("Robot initialized successfully")
        except ImportError as e:
            print(f"Robot libraries not found: {e}. Running in simulation mode.")
            self._simulation_mode = True

    def quit(self):
        if not self._simulation_mode:
            try:
                print("Moving robot to home position...")
                self.bot.arm.go_to_home_pose(moving_time=2)
                time.sleep(1)
                print("Moving robot to sleep position...")
                self.bot.arm.go_to_sleep_pose(moving_time=2)
                from interbotix_common_modules.common_robot.robot import robot_shutdown
                robot_shutdown()
                print("Robot shutdown complete")
            except Exception as e:
                print(f"Error during robot shutdown: {e}")
            
    def update_pen_heights(self, pen_down_height=0.095, pen_up_height=0.125):
        """Update pen heights with safety checks"""
        if pen_down_height < 0.05:
            print("Warning: pen_down_height too low. Setting to 0.05")
            pen_down_height = 0.05
            
        if pen_up_height - pen_down_height < 0.01:
            print("Warning: pen height difference too small. Setting pen_up_height higher.")
            pen_up_height = pen_down_height + 0.03
            
        self.pen_down_height = pen_down_height
        self.pen_up_height = pen_up_height
        self.pen_height_diff = pen_up_height - pen_down_height
        print(f"Pen heights updated: down={pen_down_height}, up={pen_up_height}")
        
    def go_to_home_pose(self):
        if not self._simulation_mode:
            try:
                self.bot.arm.go_to_home_pose(moving_time=2)
                print("Robot moved to home pose")
            except Exception as e:
                print(f"Error moving to home pose: {e}")
        else:
            print("Simulation: Going to home pose")

    def go_to_sleep_pose(self):
        if not self._simulation_mode:
            try:
                self.bot.arm.go_to_sleep_pose(moving_time=2)
                print("Robot moved to sleep pose")
            except Exception as e:
                print(f"Error moving to sleep pose: {e}")
        else:
            print("Simulation: Going to sleep pose")

    def go_to_canvas_center(self):
        if not self._simulation_mode:
            try:
                self.bot.arm.set_ee_pose_components(
                    x=self.canvas_center[0], 
                    y=self.canvas_center[1], 
                    z=self.pen_up_height, 
                    moving_time=1
                )
                print(f"Robot moved to canvas center: {self.canvas_center}")
            except Exception as e:
                print(f"Error moving to canvas center: {e}")
        else:
            print(f"Simulation: Moving to canvas center: {self.canvas_center}")

    def pen_down(self, moving_time=0.4):  # Increased moving time for smoother motion
        if not self._simulation_mode:
            try:
                self.bot.arm.set_ee_cartesian_trajectory(z=-self.pen_height_diff, moving_time=moving_time)
                # Add small delay for stability
                time.sleep(0.1)
                print("Pen down")
            except Exception as e:
                print(f"Error moving pen down: {e}")
        else:
            print("Simulation: Pen down")

    def pen_up(self, moving_time=0.4):  # Increased moving time for smoother motion
        if not self._simulation_mode:
            try:
                self.bot.arm.set_ee_cartesian_trajectory(z=self.pen_height_diff, moving_time=moving_time)
                # Add small delay for stability
                time.sleep(0.1)
                print("Pen up")
            except Exception as e:
                print(f"Error moving pen up: {e}")
        else:
            print("Simulation: Pen up")

    def gripper_close(self):
        if not self._simulation_mode:
            try:
                self.bot.gripper.grasp()
                print("Gripper closed")
            except Exception as e:
                print(f"Error closing gripper: {e}")
        else:
            print("Simulation: Gripper close")

    def gripper_open(self):
        if not self._simulation_mode:
            try:
                self.bot.gripper.release()
                print("Gripper opened")
            except Exception as e:
                print(f"Error opening gripper: {e}")
        else:
            print("Simulation: Gripper open")

    def check_canvas_bounds(self, mark_corners=False):
        """Move the robot to check the canvas bounds"""
        print("Checking canvas bounds...")
        if self._simulation_mode:
            print(f"Canvas bounds: X: {self.canvas_bounds_x}, Y: {self.canvas_bounds_y}")
            return
            
        try:
            self.go_to_canvas_center()
            time.sleep(1)
            
            bounding_points = [
                (self.canvas_bounds_x[0], self.canvas_bounds_y[0]), 
                (self.canvas_bounds_x[1], self.canvas_bounds_y[0]), 
                (self.canvas_bounds_x[1], self.canvas_bounds_y[1]), 
                (self.canvas_bounds_x[0], self.canvas_bounds_y[1])
            ]
            x0, y0 = self.canvas_center[0], self.canvas_center[1]
            for x, y in bounding_points:
                dx, dy = x - x0, y - y0
                self.bot.arm.set_ee_cartesian_trajectory(dx, dy, moving_time=0.75)
                time.sleep(1)

                if mark_corners:
                    self.pen_down()
                    time.sleep(0.5)
                    self.pen_up()
                    time.sleep(0.5)

                x0, y0 = x, y

            # Return to center
            dx, dy = self.canvas_center[0] - x0, self.canvas_center[1] - y0
            self.bot.arm.set_ee_cartesian_trajectory(dx, dy, 0, moving_time=0.75)
            time.sleep(1)
            print("Canvas bounds check complete")
        except Exception as e:
            print(f"Error checking canvas bounds: {e}")

    def draw_path(self, path):
        """Draw a path with enhanced precision and error handling"""
        if len(path) < 2:
            print("Path too short, skipping")
            return
            
        # Start at the first point
        print(f'Starting path with {len(path)} points')
        x0, y0 = path[0]
        
        # Check bounds
        if (x0 < self.canvas_bounds_x[0] or x0 > self.canvas_bounds_x[1] or 
            y0 < self.canvas_bounds_y[0] or y0 > self.canvas_bounds_y[1]):
            print(f"Warning: First point {(x0, y0)} outside canvas bounds, skipping path")
            return
            
        # Move to first point
        if not self._simulation_mode:
            try:
                self.bot.arm.set_ee_pose_components(x=x0, y=y0, z=self.pen_up_height, moving_time=1)
                time.sleep(0.3)  # Increased delay for stability
            except Exception as e:
                print(f"Error moving to first point: {e}")
                return
        else:
            print(f"Simulation: Moving to ({x0}, {y0})")

        # Pen down
        self.pen_down()

        # Draw the path with improved timing
        move_time = 0.15  # Slower movement for more precision
        for i, (x, y) in enumerate(tqdm(path[1:], desc="Drawing")):
            # Check if point is within bounds
            if (x < self.canvas_bounds_x[0] or x > self.canvas_bounds_x[1] or 
                y < self.canvas_bounds_y[0] or y > self.canvas_bounds_y[1]):
                print(f"Point {i+1} out of bounds: {(x, y)}, skipping")
                continue
                
            dx, dy = x - x0, y - y0
            dist = np.sqrt(dx**2 + dy**2)
            
            # Adjust moving time based on distance
            adjusted_time = max(0.1, move_time * (dist / 0.01))
            
            if not self._simulation_mode:
                try:
                    self.bot.arm.set_ee_cartesian_trajectory(
                        x=dx, y=dy, moving_time=adjusted_time
                    )
                    # Shorter delay for smoother motion
                    time.sleep(0.05)
                except Exception as e:
                    print(f"Error at point {i+1}: {e}")
                    break
            x0, y0 = x, y

        # Pen up
        self.pen_up()

    def convert_to_canvas_coords(self, path):
        """Convert path coordinates to canvas coordinates with bounds checking"""
        path = np.array(path)
        
        # Scale from pixels to meters
        path[:, 0] = path[:, 0] / 1000
        path[:, 1] = path[:, 1] / 1000
        
        # Translate to canvas origin
        path[:, 0] = path[:, 0] + self.origin[0]
        path[:, 1] = path[:, 1] + self.origin[1]
        
        # Safety check to ensure points are within canvas bounds
        path[:, 0] = np.clip(path[:, 0], self.canvas_bounds_x[0], self.canvas_bounds_x[1])
        path[:, 1] = np.clip(path[:, 1], self.canvas_bounds_y[0], self.canvas_bounds_y[1])
        
        return path

    def draw_image(self, image_path, debug=True):
        """Process and draw an image with enhanced path generation"""
        print("Generating improved sketch paths...")
        paths = generate_improved_sketch_paths(
            image_path, 
            target_spacing=3.0,  # Finer spacing for more detail
            size=self.canvas_size, 
            debug=debug
        )
        
        if not paths:
            print("No valid paths generated. Check image processing.")
            return
            
        # Better path sorting - longest paths first, but with some 
        # optimization to reduce travel distance
        paths = optimize_path_order(paths)
        
        print(f"Generated {len(paths)} paths")
        print(f'Canvas Bounds X: {self.canvas_bounds_x}')
        print(f'Canvas Bounds Y: {self.canvas_bounds_y}')

        self.go_to_home_pose()
        time.sleep(1)
        
        # Optional canvas bounds check
        # self.check_canvas_bounds()
        
        print("Starting drawing...")
        for i, path in enumerate(paths):
            print(f'Path {i+1}/{len(paths)}. Points: {len(path)}')
            path = self.convert_to_canvas_coords(path)
            
            if debug and i % 10 == 0:  # Show some paths during execution
                # Save figure instead of showing it to avoid Tkinter issues
                plt.figure(figsize=(6, 6))
                plt.plot(path[:, 0], path[:, 1], 'b-')
                plt.title(f"Path {i+1}")
                plt.xlim(self.canvas_bounds_x)
                plt.ylim(self.canvas_bounds_y)
                plt.savefig(f"path_{i+1}.png")
                plt.close()  # Important: close the figure to free resources
                
            self.draw_path(path)

        print("Drawing complete!")
        self.go_to_home_pose()


# Improved image processing and path planning utilities
def rgb_to_binary(image_path, blur_kernel=(7, 7), canny_thresholds=(30, 120), 
                  invert=False, resize_dim=None):
    """
    Enhanced edge detection with better noise reduction and contrast control.
    
    Parameters:
        image_path: Path to the input image
        blur_kernel: Kernel size for Gaussian blur (larger = smoother)
        canny_thresholds: Min and max thresholds for Canny edge detection
        invert: Whether to invert the image before processing (helps with some images)
        resize_dim: Optional resize dimensions (width, height)
    
    Returns:
        binary_image: 2D binary image with values 0 (background) and 1 (edges)
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if specified
    if resize_dim:
        image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Optionally invert
    if invert:
        gray = 255 - gray
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Apply Canny edge detection with custom thresholds
    edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
    
    # Convert to binary image
    binary_image = (edges > 0).astype(np.uint8)
    
    # Additional morphological operations to clean up the edges
    kernel = np.ones((2, 2), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    return binary_image, gray


def create_graph(binary_image):
    """Create a graph representation of the binary image with efficient 8-connectivity"""
    G = nx.Graph()
    
    # Add nodes to the graph
    rows, cols = binary_image.shape
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:
                G.add_node((j, i))
    
    # Add edges with 8-connectivity (more complete connections)
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:
                # Check all 8 neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # Skip self
                        
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols and 
                            binary_image[ni, nj] == 1):
                            G.add_edge((j, i), (nj, ni))
    
    return G


def extract_improved_branches(graph):
    """
    Extract branches from the graph with improved handling of junctions
    to reduce redundant paths and create more coherent strokes.
    """
    branches = []
    visited_edges = set()
    
    # Find all junction nodes (nodes with degree > 2) and endpoints (degree == 1)
    junctions = {node for node in graph.nodes() if graph.degree(node) > 2}
    endpoints = {node for node in graph.nodes() if graph.degree(node) == 1}
    
    # Start points are either junctions or endpoints
    start_points = list(junctions.union(endpoints))
    
    # Prefer endpoints for starting paths
    start_points.sort(key=lambda n: 0 if graph.degree(n) == 1 else 1)
    
    for start_node in start_points:
        neighbors = list(graph.neighbors(start_node))
        for neighbor in neighbors:
            edge = (start_node, neighbor)
            reverse_edge = (neighbor, start_node)
            
            if edge in visited_edges or reverse_edge in visited_edges:
                continue
                
            # Start a new branch
            branch = [start_node]
            current = start_node
            next_node = neighbor
            
            # Follow the path
            while True:
                # Mark the edge as visited
                visited_edges.add((current, next_node))
                visited_edges.add((next_node, current))
                
                # Add to the branch
                branch.append(next_node)
                
                # If we've reached an endpoint or junction, stop
                if graph.degree(next_node) != 2:
                    break
                    
                # Continue to the next node
                current = next_node
                # Find the unvisited neighbor
                unvisited = [n for n in graph.neighbors(current) 
                            if (current, n) not in visited_edges 
                            and n != branch[-2]]  # Avoid going back
                
                if not unvisited:
                    break
                    
                next_node = unvisited[0]
            
            # Only keep branches with at least 2 points
            if len(branch) >= 2:
                branches.append(np.array(branch))
    
    return branches


def resample_path(path, target_spacing=2.0):
    """
    Resample a path to have points spaced at a fixed distance.
    
    Parameters:
        path: np.array of shape (N, 2), original points
        target_spacing: desired spacing between points
        
    Returns:
        resampled_path: np.array of shape (M, 2), points with even spacing
    """
    if len(path) < 2:
        return path
        
    # Convert path coordinates to (x,y) format
    coords = np.array([(p[0], p[1]) for p in path])
    
    # Compute cumulative distances along the path
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_length = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = cumulative_length[-1]
    
    # If the path is too short, return as is
    if total_length < target_spacing:
        return coords
    
    # Create interpolation functions for x and y
    x_interp = interp1d(cumulative_length, coords[:, 0], kind='linear')
    y_interp = interp1d(cumulative_length, coords[:, 1], kind='linear')
    
    # Generate new points with even spacing
    num_points = max(2, int(total_length / target_spacing) + 1)
    even_distances = np.linspace(0, total_length, num_points)
    
    # Compute resampled points
    resampled_x = x_interp(even_distances)
    resampled_y = y_interp(even_distances)
    
    return np.column_stack((resampled_x, resampled_y))


def optimize_path_order(paths):
    """
    Optimize the order of paths to minimize pen travel distance.
    Combines longest-path-first logic with proximity-based ordering.
    """
    if not paths:
        return []
        
    # Sort paths by length (longest first) for initial ordering
    paths.sort(key=lambda p: len(p), reverse=True)
    
    # Take the 10 longest paths first (they define major features)
    result = paths[:10]
    remaining = paths[10:]
    
    # For remaining paths, order by proximity
    current_end = result[-1][-1]  # Last point of the last path
    
    while remaining:
        # Find the closest path to the current end point
        closest_idx = 0
        closest_dist = float('inf')
        
        for i, path in enumerate(remaining):
            # Try both the start and end of the path
            dist_to_start = np.sum((path[0] - current_end)**2)
            dist_to_end = np.sum((path[-1] - current_end)**2)
            
            min_dist = min(dist_to_start, dist_to_end)
            if min_dist < closest_dist:
                closest_dist = min_dist
                closest_idx = i
        
        # Add the closest path and remove from remaining
        next_path = remaining.pop(closest_idx)
        
        # Check if we should reverse the path
        dist_to_start = np.sum((next_path[0] - current_end)**2)
        dist_to_end = np.sum((next_path[-1] - current_end)**2)
        
        if dist_to_end < dist_to_start:
            next_path = next_path[::-1]  # Reverse the path
            
        result.append(next_path)
        current_end = next_path[-1]
    
    return result


def generate_improved_sketch_paths(image_path, target_spacing=3.0, size=None, debug=True):
    """
    Generate improved sketch paths from an image with better edge detection,
    path extraction, and path optimization.
    
    Parameters:
        image_path: Path to the input image
        target_spacing: Desired spacing between points in pixels
        size: Target size (width, height) in meters
        debug: Whether to show debug visualizations
        
    Returns:
        paths: List of np.arrays, each containing points for a stroke
    """
    # Determine appropriate resize dimensions based on target size
    resize_dim = None
    if size:
        # Convert from meters to pixels (approximately)
        resize_dim = (int(size[0] * 1000), int(size[1] * 1000))
    
    # Process the image
    binary_result, gray = rgb_to_binary(
        image_path, 
        blur_kernel=(7, 7),
        canny_thresholds=(30, 100),
        resize_dim=resize_dim
    )
    
    # Save the initial processing result instead of showing it
    if debug:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(gray, cmap='gray')
        ax1.set_title("Original Image (Grayscale)")
        
        ax2.imshow(binary_result, cmap='gray')
        ax2.set_title("Edge Detection Result")
        plt.tight_layout()
        plt.savefig("edge_detection.png")
        plt.close(fig)  # Close the figure to free resources
    
    # Create a skeleton from the binary image
    skeleton = skeletonize(binary_result)
    
    # Remove small isolated components
    skeleton = remove_small_objects(skeleton, min_size=10, connectivity=2)
    
    if debug:
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeletonized Image")
        plt.savefig("skeleton.png")
        plt.close()
    
    # Create a graph from the skeleton
    G = create_graph(skeleton)
    
    # Extract branches from the graph
    branches = extract_improved_branches(G)
    
    if not branches:
        print("No branches extracted from the image. Check image processing parameters.")
        return []
    
    # Resample the branches
    resampled_paths = []
    for branch in branches:
        if len(branch) >= 2:  # Only keep paths with at least 2 points
            resampled_path = resample_path(branch, target_spacing)
            if len(resampled_path) >= 2:
                resampled_paths.append(resampled_path)
    
    if debug:
        # Save the resampled paths instead of showing them
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton, cmap='gray', alpha=0.3)
        for path in resampled_paths:
            plt.plot(path[:, 0], path[:, 1], '-', linewidth=1)
        plt.title(f"Resampled Paths ({len(resampled_paths)} paths)")
        plt.savefig("resampled_paths.png")
        plt.close()
        
        # Print some statistics
        path_lengths = [len(path) for path in resampled_paths]
        print(f"Number of paths: {len(resampled_paths)}")
        print(f"Total points: {sum(path_lengths)}")
        print(f"Average path length: {np.mean(path_lengths):.1f} points")
        print(f"Median path length: {np.median(path_lengths):.1f} points")
        print(f"Min/Max path lengths: {min(path_lengths)}/{max(path_lengths)}")
    
    return resampled_paths


# Test and usage functions
def test_image_processing(image_path, debug=True):
    """Test the image processing pipeline without robot movement"""
    binary_result, gray = rgb_to_binary(
        image_path, 
        blur_kernel=(7, 7),
        canny_thresholds=(30, 100)
    )
    
    # Save the processing result instead of showing it
    if debug:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(gray, cmap='gray')
        ax1.set_title("Original Image (Grayscale)")
        
        ax2.imshow(binary_result, cmap='gray')
        ax2.set_title("Edge Detection Result")
        plt.tight_layout()
        plt.savefig("test_edge_detection.png")
        plt.close(fig)
    
    # Create a skeleton from the binary image
    skeleton = skeletonize(binary_result)
    
    # Remove small isolated components
    skeleton = remove_small_objects(skeleton, min_size=10, connectivity=2)
    
    if debug:
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeletonized Image")
        plt.savefig("test_skeleton.png")
        plt.close()
    
    # Create a graph from the skeleton
    G = create_graph(skeleton)
    
    # Extract branches from the graph
    branches = extract_improved_branches(G)
    
    # Resample the branches
    target_spacing = 3.0
    resampled_paths = []
    for branch in branches:
        if len(branch) >= 2:  # Only keep paths with at least 2 points
            resampled_path = resample_path(branch, target_spacing)
            if len(resampled_path) >= 2:
                resampled_paths.append(resampled_path)
    
    if debug:
        # Save the paths instead of showing them
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton, cmap='gray', alpha=0.3)
        colors = plt.cm.jet(np.linspace(0, 1, len(resampled_paths)))
        for i, path in enumerate(resampled_paths):
            plt.plot(path[:, 0], path[:, 1], '-', color=colors[i], linewidth=1)
        plt.title(f"Resampled Paths ({len(resampled_paths)} paths)")
        plt.savefig("test_resampled_paths.png")
        plt.close()
    
    return resampled_paths, skeleton


def simulate_drawing(robot, image_path):
    """Simulate the drawing process without actual robot movement"""
    # Set the robot to simulation mode
    robot._simulation_mode = True
    
    # Generate paths
    paths = generate_improved_sketch_paths(
        image_path, 
        target_spacing=3.0,
        size=robot.canvas_size, 
        debug=True
    )
    
    # Optimize path order
    paths = optimize_path_order(paths)
    
    # Plot the final drawing
    plt.figure(figsize=(10, 10))
    for path in paths:
        path = robot.convert_to_canvas_coords(path)
        plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=0.8)
    
    plt.xlim(robot.canvas_bounds_x)
    plt.ylim(robot.canvas_bounds_y)
    plt.title("Simulated Drawing")
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig("simulated_drawing.png")
    plt.close()
    
    print(f"Simulation complete with {len(paths)} paths")
    print(f"Check the output image: simulated_drawing.png")
    return paths


# Main function with robot drawing and image processing parameters
def main():
    """Main function to process and draw an image"""
    # Create a robot instance
    robot = EnhancedRobotClass()
    
    # Path to the image to draw
    image_path = 'my_picture.jpg'
    
    # Uncomment only one of these options
    
    # Option 1: Test image processing only
    # test_image_processing(image_path)
    # print("Image processing test complete. Check output images.")
    
    # Option 2: Simulate drawing
    # simulate_drawing(robot, image_path)
    # print("Simulation complete. Check 'simulated_drawing.png'")
    
    # Option 3: Actually draw the image
    robot.draw_image(image_path)
    
    # Clean up
    robot.quit()


if __name__ == "__main__":
    try:
        print("Starting robot drawing program...")
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Force cleanup matplotlib resources
        plt.close('all')
        print("Program completed.")