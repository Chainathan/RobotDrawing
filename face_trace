#!/usr/bin/env python3

import sys
import time
import os
import cv2
import random
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# FUNCTIONS DEFINITION --------------------------------------------------------------------
def rgb_to_binary(image_path, blur_kernel=(5, 5), canny_thresholds=(50, 150)):
    """
    Converts an RGB image into a binary image, retaining all essential features.
    Returns None if the image cannot be loaded.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
        edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
        # Multiply by 255 to create a proper binary image for contour extraction
        binary_image = (edges > 0).astype(np.uint8) * 255
        return binary_image
    except Exception as e:
        print(f"Error in rgb_to_binary: {e}")
        return None


def tsp_nearest_neighbor(points):
    """Solve TSP using a nearest neighbor heuristic."""
    if len(points) < 2:
        return points
    points = [tuple(p) for p in points]
    path = [points[0]]
    remaining = set(points[1:])
    while remaining:
        last = path[-1]
        nearest = min(remaining, key=lambda p: np.linalg.norm(np.array(last) - np.array(p)))
        path.append(nearest)
        remaining.remove(nearest)
    return path


def perpendicular_distance(point, start, end):
    """Compute the perpendicular distance from a point to a line segment (start-end)."""
    if start == end:
        return np.linalg.norm(np.array(point) - np.array(start))
    x0, y0 = point
    x1, y1 = start
    x2, y2 = end
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return num / den


def douglas_peucker(points, epsilon):
    """Simplify the path using the Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    max_dist = 0
    index = 0
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i
    if max_dist > epsilon:
        left = douglas_peucker(points[:index + 1], epsilon)
        right = douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def generate_smooth_path(x_points, y_points, num_interp=500):
    """Generates a smooth path using cubic interpolation (Catmull-Rom style)."""
    t = np.linspace(0, 1, len(x_points))
    t_interp = np.linspace(0, 1, num_interp)
    interp_x = interp1d(t, x_points, kind='cubic')
    interp_y = interp1d(t, y_points, kind='cubic')
    smooth_x = interp_x(t_interp)
    smooth_y = interp_y(t_interp)
    return smooth_x, smooth_y


def extract_face_path(image_path):
    """Extracts face contours and generates a smooth path for the robotic arm."""
    binary_result = rgb_to_binary(image_path)
    if binary_result is None:
        print("Failed to process the image. Exiting.")
        return None

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_points = [tuple(pt[0]) for contour in contours for pt in contour]
    if len(edge_points) == 0:
        print("No contours found in the image.")
        return None

    # Sample points for clustering
    num_samples = min(3000, len(edge_points))
    sampled_nodes = random.sample(edge_points, num_samples)
    node_array = np.array(sampled_nodes)

    # Cluster points using DBSCAN
    eps = 5
    min_samples = 2
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(node_array)
    cluster_labels = db.labels_

    # Organize points into clusters
    clusters = {}
    for point, label in zip(sampled_nodes, cluster_labels):
        if label != -1:
            clusters.setdefault(label, []).append(point)

    # Solve TSP for each cluster and simplify the path
    cluster_paths = {}
    for label, cluster_points in clusters.items():
        tsp_path = tsp_nearest_neighbor(cluster_points)
        cluster_paths[label] = douglas_peucker(tsp_path, epsilon=1.5)

    return cluster_paths

# FUNCTIONS DEFINITION ENDS----------------------------------------------------------------

def main():
    # Initialize the robot interface
    bot = InterbotixManipulatorXS(
        robot_model='wx250s',
        group_name='arm',
        gripper_name='gripper',
    )
    robot_startup()

    # Set a safe starting pose (XY plane with constant Z)
    start_x = 0.25
    start_y = 0.0
    start_z = 0.1
    bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=start_z)
    time.sleep(2)  # Allow time to reach starting pose

    # Image path and verification
    image_path = os.path.expanduser("~/Downloads/my_picture.jpg")  # Update this path to your image location
    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}. Exiting.")
        robot_shutdown()
        return

    # Load image to get dimensions for coordinate transformation
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Extract face path from the image
    cluster_paths = extract_face_path(image_path)
    if cluster_paths is None:
        robot_shutdown()
        return

    # Define scaling factor to convert pixel coordinates to meters (robot workspace)
    scale = 0.001  # Adjust as needed

    # Draw the face pattern for each cluster
    for label, path in cluster_paths.items():
        x_points = [p[0] for p in path]
        y_points = [p[1] for p in path]
        if len(x_points) < 2:
            continue

        # Smooth the path using interpolation
        smoothed_x, smoothed_y = generate_smooth_path(x_points, y_points, num_interp=500)

        # Transform image coordinates to robot workspace coordinates.
        # Here, the image center is mapped to the starting pose.
        transformed_path = []
        for x, y in zip(smoothed_x, smoothed_y):
            robot_x = start_x + (x - width/2) * scale
            robot_y = start_y + (y - height/2) * scale
            transformed_path.append((robot_x, robot_y))

        # Move to the start of the transformed path
        initial_x, initial_y = transformed_path[0]
        bot.arm.set_ee_pose_components(x=initial_x, y=initial_y, z=start_z)
        time.sleep(1)
        current_x, current_y = initial_x, initial_y

        # Move the robot along the transformed (and smoothed) path
        for robot_x, robot_y in transformed_path:
            dx, dy = robot_x - current_x, robot_y - current_y
            bot.arm.set_ee_cartesian_trajectory(x=dx, y=dy, moving_time=0.08)
            current_x, current_y = robot_x, robot_y
            time.sleep(0.1)  # Adjust timing as needed for smooth execution

    # Return to the starting position before shutting down
    bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=start_z)
    time.sleep(1)
    bot.arm.go_to_sleep_pose()

    # Shutdown the robot
    robot_shutdown()


if __name__ == '__main__':
    main()
