# Corrected Facial Feature Enhanced Robot Drawing
# With anti-stuck mechanisms and improved performance
# ====================================

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
from collections import deque, OrderedDict
import time
from tqdm import tqdm

class FacialFeatureEnhancedRobot:
    def __init__(self):
        '''
        Enhanced robot class with special handling for facial features
        and anti-stuck mechanisms
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

        # Anti-stuck parameters
        self.movement_timeout = 3.0    # Maximum seconds to wait for a movement
        self.max_retries = 2           # Number of retries for a movement
        self.stuck_counter = 0         # Track number of times robot gets stuck

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

    def pen_down(self, moving_time=0.5):  # Increased time for more reliable pen down
        if not self._simulation_mode:
            try:
                self.bot.arm.set_ee_cartesian_trajectory(z=-self.pen_height_diff, moving_time=moving_time)
                # Add small delay for stability
                time.sleep(0.2)
                print("Pen down")
            except Exception as e:
                print(f"Error moving pen down: {e}")
        else:
            print("Simulation: Pen down")

    def pen_up(self, moving_time=0.5):  # Increased time for more reliable pen up
        if not self._simulation_mode:
            try:
                self.bot.arm.set_ee_cartesian_trajectory(z=self.pen_height_diff, moving_time=moving_time)
                # Add small delay for stability
                time.sleep(0.2)
                print("Pen up")
            except Exception as e:
                print(f"Error moving pen up: {e}")
        else:
            print("Simulation: Pen up")

    def draw_path(self, path, is_facial_feature=False):
        """
        Draw a path with enhanced precision, error handling, and anti-stuck mechanisms.
        For facial features, use slower, more precise movements.
        """
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
                # Use longer moving time for the initial positioning
                self.bot.arm.set_ee_pose_components(x=x0, y=y0, z=self.pen_up_height, moving_time=1.5)
                time.sleep(0.5)  # Increased delay for stability
            except Exception as e:
                print(f"Error moving to first point: {e}")
                return
        else:
            print(f"Simulation: Moving to ({x0}, {y0})")

        # Pen down
        self.pen_down()

        # Reset stuck counter for new path
        self.stuck_counter = 0

        # Adjust movement time based on whether this is a facial feature
        # IMPORTANT: Increased base times to reduce sticking
        base_move_time = 0.3 if is_facial_feature else 0.25  # Increased from 0.2/0.15

        # Draw the path with improved timing and anti-stuck mechanisms
        skipped_points = 0
        for i, (x, y) in enumerate(tqdm(path[1:], desc="Drawing")):
            # Check if point is within bounds
            if (x < self.canvas_bounds_x[0] or x > self.canvas_bounds_x[1] or 
                y < self.canvas_bounds_y[0] or y > self.canvas_bounds_y[1]):
                print(f"Point {i+1} out of bounds: {(x, y)}, skipping")
                skipped_points += 1
                continue
                
            # Check if we're skipping too many points (path might be problematic)
            if skipped_points > len(path) * 0.3:  # If more than 30% of points are skipped
                print(f"Too many points skipped ({skipped_points}), abandoning path")
                break
                
            dx, dy = x - x0, y - y0
            dist = np.sqrt(dx**2 + dy**2)
            
            # Skip extremely small movements that might cause sticking
            if dist < 0.001:  # 1mm minimum movement
                # print(f"Movement too small ({dist:.5f}m), skipping point {i+1}")
                continue
            
            # Adjust moving time based on distance and whether this is a facial feature
            # IMPORTANT: Increased minimum time from 0.1 to 0.3 seconds
            adjusted_time = max(0.3, base_move_time * (dist / 0.01))
            
            if not self._simulation_mode:
                success = False
                retries = 0
                
                while not success and retries < self.max_retries:
                    try:
                        # Record start time for timeout detection
                        start_time = time.time()
                        
                        # Execute the movement
                        self.bot.arm.set_ee_cartesian_trajectory(
                            x=dx, y=dy, moving_time=adjusted_time
                        )
                        
                        # Check if movement took too long (might be stuck)
                        elapsed = time.time() - start_time
                        if elapsed > self.movement_timeout:
                            print(f"Movement timed out ({elapsed:.2f}s), point {i+1}")
                            self.stuck_counter += 1
                            
                            # If robot gets stuck too often, try a small Z movement to unstick
                            if self.stuck_counter > 3:
                                print("Robot appears stuck, attempting recovery...")
                                try:
                                    # Small up and down movement to break sticking
                                    self.bot.arm.set_ee_cartesian_trajectory(z=0.005, moving_time=0.3)
                                    time.sleep(0.1)
                                    self.bot.arm.set_ee_cartesian_trajectory(z=-0.005, moving_time=0.3)
                                    time.sleep(0.1)
                                    self.stuck_counter = 0  # Reset counter after recovery
                                except:
                                    pass
                            
                            # Skip to next point
                            break
                        
                        # Short delay for stability
                        time.sleep(0.05)
                        success = True
                        
                    except Exception as e:
                        print(f"Error at point {i+1}: {e}")
                        retries += 1
                        time.sleep(0.1)  # Small delay before retry
                
                if not success:
                    print(f"Failed to move to point {i+1} after {self.max_retries} attempts, skipping")
                    continue
            
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
        """Process and draw an image with enhanced facial feature detection"""
        print("Generating sketch paths with facial feature enhancement...")
        
        # Process the image with facial feature detection
        face_feature_paths, regular_paths = generate_face_enhanced_paths(
            image_path, 
            target_spacing=3.5,  # Increased spacing from 3.0 to 3.5
            size=self.canvas_size, 
            debug=debug
        )
        
        if not face_feature_paths and not regular_paths:
            print("No valid paths generated. Check image processing.")
            return
            
        # Optimize path order for regular paths
        regular_paths = optimize_path_order(regular_paths)
        
        # Simplify paths to reduce sticking
        simplified_feature_paths = []
        for feature_type, path in face_feature_paths:
            # Only keep longer facial feature paths (more important details)
            if len(path) > 5:
                # Additional simplification for very long paths
                if len(path) > 30:
                    # Subsample very long paths to reduce points
                    indices = np.linspace(0, len(path)-1, min(30, len(path))).astype(int)
                    path = path[indices]
                simplified_feature_paths.append((feature_type, path))
        
        simplified_regular_paths = []
        for path in regular_paths:
            # Only keep longer regular paths
            if len(path) > 5:
                # Additional simplification for very long paths
                if len(path) > 30:
                    # Subsample very long paths to reduce points
                    indices = np.linspace(0, len(path)-1, min(30, len(path))).astype(int)
                    path = path[indices]
                simplified_regular_paths.append(path)
        
        print(f"Generated {len(simplified_feature_paths)} facial feature paths and {len(simplified_regular_paths)} regular paths")
        print(f"(Simplified from {len(face_feature_paths)} and {len(regular_paths)} original paths)")
        print(f'Canvas Bounds X: {self.canvas_bounds_x}')
        print(f'Canvas Bounds Y: {self.canvas_bounds_y}')

        self.go_to_home_pose()
        time.sleep(1)
        
        # First draw facial features with more precision
        if simplified_feature_paths:
            print("Drawing facial features...")
            
            # Sort facial feature paths by importance
            feature_order = {"right_eye": 0, "left_eye": 1, "nose": 2, "mouth": 3, 
                           "right_eyebrow": 4, "left_eyebrow": 5, "jaw": 6}
            simplified_feature_paths.sort(key=lambda x: feature_order.get(x[0], 99))
            
            for i, (feature_type, path) in enumerate(simplified_feature_paths):
                print(f'Drawing {feature_type} - Path {i+1}/{len(simplified_feature_paths)}')
                canvas_path = self.convert_to_canvas_coords(path)
                
                if debug:
                    # Save figure instead of showing it
                    plt.figure(figsize=(6, 6))
                    plt.plot(canvas_path[:, 0], canvas_path[:, 1], 'r-')
                    plt.title(f"Facial Feature: {feature_type}")
                    plt.xlim(self.canvas_bounds_x)
                    plt.ylim(self.canvas_bounds_y)
                    plt.savefig(f"facial_feature_{feature_type}_{i}.png")
                    plt.close()
                    
                # Draw facial features with extra precision
                self.draw_path(canvas_path, is_facial_feature=True)
                
                # Add small delay between facial features
                time.sleep(0.5)
        
        # Then draw regular paths
        print("Drawing remaining paths...")
        for i, path in enumerate(simplified_regular_paths):
            print(f'Path {i+1}/{len(simplified_regular_paths)}. Points: {len(path)}')
            canvas_path = self.convert_to_canvas_coords(path)
            
            if debug and i % 10 == 0:
                # Save figure instead of showing it
                plt.figure(figsize=(6, 6))
                plt.plot(canvas_path[:, 0], canvas_path[:, 1], 'b-')
                plt.title(f"Regular Path {i+1}")
                plt.xlim(self.canvas_bounds_x)
                plt.ylim(self.canvas_bounds_y)
                plt.savefig(f"path_{i+1}.png")
                plt.close()
                
            self.draw_path(canvas_path)
            
            # Small rest between paths to prevent overheating/stressing the motors
            if i % 5 == 0:
                time.sleep(0.2)

        print("Drawing complete!")
        self.go_to_home_pose()


# Facial landmark detection functions
def detect_facial_landmarks(image):
    """
    Detect facial landmarks in the image.
    
    Returns:
        landmarks: List of (x, y) coordinates for the 68 facial landmarks
        face_rect: Rectangle (x, y, w, h) containing the face
    """
    # Initialize dlib's face detector and facial landmark predictor
    try:
        import dlib
        
        # Initialize face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        
        # Check if shape predictor file exists, download if not
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Facial landmark predictor file not found.")
            print("Please download it from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract and place in the current directory")
            return None, None
            
        predictor = dlib.shape_predictor(predictor_path)
        
        # Detect faces in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return None, None
            
        # Get the first face (assuming portrait has one main face)
        face = faces[0]
        face_rect = (face.left(), face.top(), face.width(), face.height())
        
        # Predict facial landmarks
        shape = predictor(gray, face)
        
        # Convert landmarks to numpy array
        landmarks = []
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            landmarks.append((x, y))
            
        return landmarks, face_rect
        
    except ImportError:
        print("Dlib not installed. Facial feature enhancement not available.")
        print("Install dlib with: pip install dlib")
        return None, None
    except Exception as e:
        print(f"Error detecting facial landmarks: {e}")
        return None, None


def extract_facial_feature_regions(image, landmarks, debug=False):
    """
    Extract regions for different facial features.
    
    Returns:
        regions: Dict of feature name to masked image region
    """
    if landmarks is None:
        return {}
        
    # Define the facial landmarks indices for each facial feature
    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 35)),
        ("jaw", (0, 17))
    ])
    
    # Create a copy of the image
    regions = {}
    
    # Extract each facial feature
    for name, (start, end) in FACIAL_LANDMARKS_IDXS.items():
        # Get the landmarks for this facial feature
        points = np.array(landmarks[start:end])
        
        if len(points) > 0:
            # Create a mask for this feature
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # Fill the convex polygon (for eyes, mouth, nose)
            if name in ["mouth", "right_eye", "left_eye", "nose"]:
                cv2.fillConvexPoly(mask, points, 255)
                
                # Dilate the mask to include area around the feature
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                # For other features, draw lines to connect the points
                for i in range(len(points) - 1):
                    cv2.line(mask, tuple(points[i]), tuple(points[i + 1]), 255, 2)
                
                # Dilate the lines to make them more visible
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Save the masked region
            regions[name] = mask
            
            if debug:
                # Save the mask
                cv2.imwrite(f"{name}_mask.png", mask)
    
    return regions


def process_facial_feature(image, feature_mask, feature_name, debug=False):
    """
    Process a facial feature to extract detailed paths.
    
    Returns:
        paths: List of paths for this facial feature
    """
    # Apply the mask to the original image
    masked_feature = cv2.bitwise_and(image, image, mask=feature_mask)
    
    # Convert to grayscale
    gray = cv2.cvtColor(masked_feature, cv2.COLOR_BGR2GRAY)
    
    # Apply appropriate edge detection based on feature type
    if feature_name in ["right_eye", "left_eye"]:
        # For eyes, use lower thresholds to capture more detail
        edges = cv2.Canny(gray, 20, 80)
    elif feature_name == "nose":
        # For nose, use higher thresholds to get main structure
        edges = cv2.Canny(gray, 30, 90)
    elif feature_name == "mouth":
        # For mouth, focus on lips outline
        edges = cv2.Canny(gray, 25, 85)
    else:
        # Default parameters for other features
        edges = cv2.Canny(gray, 30, 100)
    
    # Create a skeleton from the edges
    binary = (edges > 0).astype(np.uint8)
    skeleton = skeletonize(binary)
    
    # Remove small isolated components
    skeleton = remove_small_objects(skeleton, min_size=5, connectivity=2)
    
    if debug:
        # Save the processed feature
        cv2.imwrite(f"{feature_name}_edges.png", edges)
        plt.figure(figsize=(6, 6))
        plt.imshow(skeleton, cmap='gray')
        plt.title(f"{feature_name} Skeleton")
        plt.savefig(f"{feature_name}_skeleton.png")
        plt.close()
    
    # Create a graph from the skeleton
    G = create_graph(skeleton)
    
    # Extract branches
    branches = extract_improved_branches(G)
    
    # Resample the branches with appropriate spacing
    # IMPORTANT: Increased spacing for facial features from 2.0 to 3.0
    target_spacing = 3.0  # Increased spacing for facial features
    
    resampled_paths = []
    for branch in branches:
        if len(branch) >= 2:
            resampled_path = resample_path(branch, target_spacing)
            if len(resampled_path) >= 2:
                resampled_paths.append(resampled_path)
    
    return resampled_paths


def generate_face_enhanced_paths(image_path, target_spacing=3.5, size=None, debug=True):
    """
    Generate drawing paths with enhanced facial feature detection.
    
    Returns:
        facial_feature_paths: List of (feature_name, path) tuples
        regular_paths: List of paths for non-facial regions
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
        
    # Resize if specified
    if size:
        resize_dim = (int(size[0] * 1000), int(size[1] * 1000))
        image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)
    
    # Detect facial landmarks
    landmarks, face_rect = detect_facial_landmarks(image)
    
    facial_feature_paths = []
    
    # If landmarks were detected, process facial features separately
    if landmarks:
        # Extract facial feature regions
        feature_regions = extract_facial_feature_regions(image, landmarks, debug)
        
        # Process each facial feature
        for feature_name, feature_mask in feature_regions.items():
            feature_paths = process_facial_feature(image, feature_mask, feature_name, debug)
            
            # Add feature paths with their names
            for path in feature_paths:
                facial_feature_paths.append((feature_name, path))
                
        if debug:
            # Draw the landmarks on the image
            img_with_landmarks = image.copy()
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(img_with_landmarks, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(img_with_landmarks, str(i), (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            cv2.imwrite("facial_landmarks.png", img_with_landmarks)
            
            # Create a visualization of all facial features
            face_features_viz = image.copy()
            colors = {
                "mouth": (0, 0, 255),    # Red
                "right_eye": (0, 255, 0), # Green
                "left_eye": (0, 255, 0),  # Green
                "nose": (255, 0, 0),      # Blue
                "right_eyebrow": (255, 255, 0), # Cyan
                "left_eyebrow": (255, 255, 0),  # Cyan
                "jaw": (0, 255, 255)      # Yellow
            }
            
            for feature_name, feature_mask in feature_regions.items():
                # Create a colored mask for visualization
                color_mask = np.zeros_like(image)
                color_mask[feature_mask > 0] = colors.get(feature_name, (255, 255, 255))
                
                # Blend with original image
                alpha = 0.5
                cv2.addWeighted(color_mask, alpha, face_features_viz, 1-alpha, 0, face_features_viz)
            
            cv2.imwrite("facial_features.png", face_features_viz)
    
    # Generate paths for the whole image (for non-facial regions)
    # Create a mask for non-facial regions if landmarks were detected
    non_facial_mask = None
    if landmarks and face_rect:
        # Create a mask for the face region
        non_facial_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Exclude facial feature regions from the mask
        for _, feature_mask in feature_regions.items():
            non_facial_mask[feature_mask > 0] = 0
    
    # Process the image for regular paths
    if non_facial_mask is not None:
        # Apply the non-facial mask
        non_facial_image = cv2.bitwise_and(image, image, mask=non_facial_mask)
        gray = cv2.cvtColor(non_facial_image, cv2.COLOR_BGR2GRAY)
    else:
        # Process the entire image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Regular edge detection process
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Create binary image
    binary_result = (edges > 0).astype(np.uint8)
    
    if debug:
        # Save processed images
        cv2.imwrite("gray.png", gray)
        cv2.imwrite("edges.png", edges)
        
        if non_facial_mask is not None:
            cv2.imwrite("non_facial_mask.png", non_facial_mask)
    
    # Create skeleton
    skeleton = skeletonize(binary_result)
    
    # Remove small isolated components
    skeleton = remove_small_objects(skeleton, min_size=10, connectivity=2)
    
    if debug:
        # Save the skeleton
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeletonized Image")
        plt.savefig("skeleton.png")
        plt.close()
    
    # Create a graph from the skeleton
    G = create_graph(skeleton)
    
    # Extract branches
    branches = extract_improved_branches(G)
    
    # Resample the branches
    regular_paths = []
    for branch in branches:
        if len(branch) >= 2:
            resampled_path = resample_path(branch, target_spacing)
            if len(resampled_path) >= 2:
                regular_paths.append(resampled_path)
    
    if debug:
        # Plot facial feature paths and regular paths
        plt.figure(figsize=(10, 10))
        
        # Draw facial feature paths in different colors
        feature_colors = {
            "mouth": 'r',
            "right_eye": 'g',
            "left_eye": 'g',
            "nose": 'b',
            "right_eyebrow": 'c',
            "left_eyebrow": 'c',
            "jaw": 'y'
        }
        
        for feature_name, path in facial_feature_paths:
            color = feature_colors.get(feature_name, 'm')
            plt.plot(path[:, 0], path[:, 1], color, linewidth=1)
        
        # Draw regular paths
        for path in regular_paths:
            plt.plot(path[:, 0], path[:, 1], 'k-', linewidth=0.5, alpha=0.7)
        
        plt.title("All Paths (Green=Eyes, Red=Mouth, Blue=Nose, Cyan=Eyebrows, Yellow=Jaw, Black=Other)")
        plt.gca().set_aspect('equal')
        plt.savefig("all_paths.png")
        plt.close()
        
        # Print statistics
        print(f"Facial feature paths: {len(facial_feature_paths)}")
        for feature in set(name for name, _ in facial_feature_paths):
            count = sum(1 for name, _ in facial_feature_paths if name == feature)
            print(f"  - {feature}: {count} paths")
        
        print(f"Regular paths: {len(regular_paths)}")
    
    return facial_feature_paths, regular_paths


# Utility functions for path processing
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
    # Limit the number of points to prevent very dense paths
    num_points = min(num_points, 30)
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
    result = paths[:min(10, len(paths))]
    remaining = paths[min(10, len(paths)):]
    
    # For remaining paths, order by proximity
    if result:
        current_end = result[-1][-1]  # Last point of the last path
    else:
        return []
    
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


# Main script to run the facial feature enhanced portrait drawing
def main():
    """Main function to process and draw a portrait with facial feature enhancement"""
    # Create a robot instance
    robot = FacialFeatureEnhancedRobot()
    
    # Path to the image to draw
    image_path = 'my_picture.jpg'
    
    # Draw the image with facial feature enhancement
    robot.draw_image(image_path)
    
    # Clean up
    robot.quit()


if __name__ == "__main__":
    try:
        print("Starting facial feature enhanced drawing...")
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Force cleanup matplotlib resources
        plt.close('all')
        print("Program completed.")