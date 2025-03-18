#!/usr/bin/env python3
import time
import os
import cv2
import numpy as np
from scipy.interpolate import interp1d
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

def extract_contour_paths(image_path):
    """Extracts contours from the given image to create drawing paths."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Failed to load image.")
        return []

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            paths.append([tuple(pt[0]) for pt in contour])

    return paths

def generate_smooth_path(points, num_interp=50):
    """Generates a smooth path using cubic interpolation."""
    if len(points) < 2:
        return points

    x_points, y_points = zip(*points)
    t = np.linspace(0, 1, len(points))
    t_interp = np.linspace(0, 1, num_interp)

    interp_x = interp1d(t, x_points, kind='cubic', fill_value="extrapolate")
    interp_y = interp1d(t, y_points, kind='cubic', fill_value="extrapolate")

    smooth_x = interp_x(t_interp)
    smooth_y = interp_y(t_interp)
    
    return list(zip(smooth_x, smooth_y))

def main():
    """Main function to initialize the robot and draw a simple shape."""
    # Initialize the robot
    bot = InterbotixManipulatorXS(
        robot_model='wx250s',
        group_name='arm',
        gripper_name='gripper',
    )
    robot_startup()

    start_x, start_y, start_z = 0.25, 0.0, 0.1
    bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=start_z)
    time.sleep(1)

    # Define a simple square path
    square_size = 0.1  # meters
    square_path = [
        (0, 0),
        (square_size, 0),
        (square_size, square_size),
        (0, square_size),
        (0, 0)
    ]

    # Generate a smooth path for the square
    smooth_square_path = generate_smooth_path(square_path, num_interp=50)

    # Draw the square
    robot_moving_time = 0.05
    pause_time = 0.01  # Reduced pause time for faster execution

    current_x, current_y = smooth_square_path[0]
    bot.arm.set_ee_pose_components(x=current_x, y=current_y, z=start_z)
    time.sleep(0.1)

    for robot_x, robot_y in smooth_square_path[1:]:
        dx = robot_x - current_x
        dy = robot_y - current_y
        bot.arm.set_ee_cartesian_trajectory(x=dx, y=dy, moving_time=robot_moving_time)
        current_x, current_y = robot_x, robot_y
        time.sleep(pause_time)

    bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=start_z)
    time.sleep(0.1)
    bot.arm.go_to_sleep_pose()
    robot_shutdown()

if __name__ == '__main__':
    main()
