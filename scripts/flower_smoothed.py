#!/usr/bin/env python3

import sys
import time
import numpy as np
from scipy.interpolate import interp1d  # Built-in interpolation (no need for bezier)
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

def generate_smooth_path(x_points, y_points, num_interp=500):
    """ Generates a smooth path using cubic interpolation (Catmull-Rom style) """
    
    # Create a parametric index for interpolation
    t = np.linspace(0, 1, len(x_points))  # Parameterized time points
    t_interp = np.linspace(0, 1, num_interp)  # More points for a smooth curve

    # Interpolate using cubic splines
    interp_x = interp1d(t, x_points, kind='cubic')
    interp_y = interp1d(t, y_points, kind='cubic')

    # Generate smooth interpolated points
    smooth_x = interp_x(t_interp)
    smooth_y = interp_y(t_interp)

    return smooth_x, smooth_y

def main():
    # Initialize the robot interface
    bot = InterbotixManipulatorXS(
        robot_model='wx250s',
        group_name='arm',
        gripper_name='gripper',  # Gripper not used in this demo
    )

    robot_startup()

    # Set a safe starting pose (XY plane with constant Z)
    start_x = 0.25
    start_y = 0.0
    start_z = 0.1
    bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=start_z)
    time.sleep(2)  # Allow time to reach starting pose

    # Define the rose curve parameters
    num_points = 100  # Fewer points before smoothing
    a = 0.08          # Scaled-up flower
    k = 6             # 6-petal rose

    # Generate the rough rose curve points
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = a * np.cos(k * theta)
    x_points = start_x + r * np.cos(theta)
    y_points = start_y + r * np.sin(theta)

    # Smooth the curve using interpolation
    smoothed_x, smoothed_y = generate_smooth_path(x_points, y_points, num_interp=500)

    # Draw the AI-smoothed flower pattern
    current_x, current_y = start_x, start_y
    for x, y in zip(smoothed_x, smoothed_y):
        dx, dy = x - current_x, y - current_y
        bot.arm.set_ee_cartesian_trajectory(x=dx, y=dy, moving_time=0.08)
        current_x, current_y = x, y
        time.sleep(0.1)  # Optimized timing for smooth execution

    # Move back to starting position before shutting down
    bot.arm.set_ee_pose_components(x=start_x, y=start_y, z=start_z)
    time.sleep(1)

    # Transition smoothly to the sleep pose
    bot.arm.go_to_sleep_pose()

    robot_shutdown()

if __name__ == '__main__':
    main()
