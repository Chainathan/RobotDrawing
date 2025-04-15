import os
import cv2
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time
from tqdm import tqdm
from path_planning_utils import generate_sketch_paths

class RobotClass:
    """
    RobotClass encapsulates the control logic for the Trossen WidowX 250s robotic arm 
    to perform sketching tasks on a fixed canvas surface.

    This class provides a complete interface for:
        - Initializing robot connection and drawing workspace
        - Managing pen up/down heights for simulating pen motion
        - Converting image paths to canvas-space trajectories
        - Executing smooth drawing operations via Cartesian movements
        - Safely starting up and shutting down the robot

    Typical Usage:
    --------------
    from RobotClass import RobotClass

    robot = RobotClass()
    robot.draw_image("path/to/image.jpg")  # Draws a sketch version of the image
    robot.quit()  # Safely returns robot to sleep position and shuts it down

    Key Parameters:
    ---------------
    - pen_up_height: Z-height when the pen is lifted
    - pen_down_height: Z-height when the pen is touching the surface
    - canvas_center: Real-world center of the drawing area (in meters)
    - canvas_size: Physical size of the drawable region (in meters)

    Main Methods:
    -------------
    - draw_image(image_path): Executes full pipeline — image → sketch → robot path
    - draw_path(path): Executes one continuous stroke on the canvas
    - convert_to_canvas_coords(path): Maps (x, y) pixels to real-world robot coords
    - pen_up() / pen_down(): Simulates lifting or placing the pen
    - go_to_home_pose(), go_to_sleep_pose(): Moves robot to predefined poses
    - check_canvas_bounds(): (Optional) Diagnostic for physical canvas area
    - update_pen_heights(): Adjusts pen heights based on calibration

    Dependencies:
    -------------
    - Interbotix ROS packages
    - path_planning_utils.py for sketch generation
    - NumPy, OpenCV, NetworkX, tqdm

    Notes:
    ------
    - All units internally are in meters
    - Use `generate_sketch_paths()` to convert images to trajectory paths
    - This class assumes a fixed drawing surface and pre-calibrated pen mount
    """
    def __init__(self):
        grid_size = 250 / 1000
        side = 0.25
        self.canvas_size = (side, side)
        self.canvas_center = (0.375, 0)
        self.canvas_bounds_x = (self.canvas_center[0] - side/2, self.canvas_center[0] + side/2)
        self.canvas_bounds_y = (self.canvas_center[1] - side/2, self.canvas_center[1] + side/2)
        # self.origin = (self.canvas_bounds_y[0], self.canvas_bounds_x[0])
        self.origin = (self.canvas_bounds_x[0], self.canvas_bounds_y[0])

        # HEIGHT PARAMS
        self.pen_up_height = 0.125
        self.pen_down_height = 0.1
        self.pen_height_diff = self.pen_up_height - self.pen_down_height

        # Initialize the robot interface
        self.bot = InterbotixManipulatorXS(
            robot_model='wx250s',
            group_name='arm',
            gripper_name='gripper',
        )

        robot_startup()

    def quit(self):
        self.bot.arm.go_to_home_pose(moving_time=2)
        time.sleep(1)
        self.bot.arm.go_to_sleep_pose(moving_time=2)
        robot_shutdown()

    def update_pen_heights(self, pen_down_height=0.07, pen_up_height=0.11):
        self.pen_down_height = pen_down_height
        self.pen_up_height = pen_up_height
        self.pen_height_diff = pen_up_height - pen_down_height

    def go_to_home_pose(self):
        self.bot.arm.go_to_home_pose(moving_time=2)

    def go_to_sleep_pose(self):
        self.bot.arm.go_to_sleep_pose(moving_time=2)

    def go_to_canvas_center(self):
        self.bot.arm.set_ee_pose_components(x=self.canvas_center[0], y=self.canvas_center[1], z=self.pen_up_height, moving_time=1)

    def pen_down(self, moving_time=0.2):
        self.bot.arm.set_ee_cartesian_trajectory(z=-self.pen_height_diff, moving_time=moving_time)

    def pen_up(self, moving_time=0.2):
        self.bot.arm.set_ee_cartesian_trajectory(z=self.pen_height_diff, moving_time=moving_time)

    def gripper_close(self):
        self.bot.gripper.grasp()

    def gripper_open(self):
        self.bot.gripper.release()

    def check_canvas_bounds(self, mark_corners=False):
        # Move the robot to the canvas bounds
        self.bot.arm.set_ee_pose_components(x=self.canvas_center[0], y=self.canvas_center[1], z=self.pen_up_height, moving_time=1)
        time.sleep(1)
        
        bounding_points = [(self.canvas_bounds_x[0], self.canvas_bounds_y[0]), 
                        (self.canvas_bounds_x[1], self.canvas_bounds_y[0]), 
                        (self.canvas_bounds_x[1], self.canvas_bounds_y[1]), 
                        (self.canvas_bounds_x[0], self.canvas_bounds_y[1])]
        x0, y0 = self.canvas_center[0], self.canvas_center[1]
        for x, y in bounding_points:
            dx, dy = x - x0, y - y0
            self.bot.arm.set_ee_cartesian_trajectory(dx, dy, moving_time=0.75)
            time.sleep(1)

            if mark_corners:
                self.bot.arm.set_ee_cartesian_trajectory(z=-self.pen_height_diff, moving_time=0.75)
                time.sleep(1)
                self.bot.arm.set_ee_cartesian_trajectory(z=self.pen_height_diff, moving_time=0.75)
                time.sleep(1)

            x0, y0 = x, y

        dx, dy = self.canvas_center[0] - x0, self.canvas_center[1] - y0
        self.bot.arm.set_ee_cartesian_trajectory(dx, dy, 0, moving_time=0.75)
        time.sleep(1)

    def check_pen_height_pos(self):
        self.bot.arm.set_ee_pose_components(x=self.canvas_center[0], y=self.canvas_center[1], z=self.pen_up_height, moving_time=1)
        time.sleep(0.5)

        self.bot.arm.set_ee_pose_components(x=self.canvas_center[0], y=self.canvas_center[1], z=self.pen_down_height, moving_time=1)
        time.sleep(0.5)

        self.bot.arm.set_ee_pose_components(x=self.canvas_center[0], y=self.canvas_center[1], z=self.pen_up_height, moving_time=1)
        time.sleep(0.5)

        self.bot.arm.set_ee_pose_components(x=self.canvas_center[0], y=self.canvas_center[1], z=self.pen_down_height, moving_time=1)
        time.sleep(0.5)

        self.bot.arm.set_ee_pose_components(x=self.canvas_center[0], y=self.canvas_center[1], z=self.pen_up_height, moving_time=1)
        time.sleep(0.5)

    def draw_path(self, path):
        # Start at the first point
        print('First point:', path[0])
        x0, y0 = path[0]
        self.bot.arm.set_ee_pose_components(x=x0, y=y0, z=self.pen_up_height, moving_time=1)
        time.sleep(1)

        print('Pen down')
        # Pen down
        self.bot.arm.set_ee_cartesian_trajectory(z=-self.pen_height_diff, moving_time=0.3)
        time.sleep(0.1)

        print('Drawing path...')
        # Draw the path
        for x, y in tqdm(path[1:], desc="Path Coords"):
            dx, dy = x - x0, y - y0
            self.bot.arm.set_ee_cartesian_trajectory(x=dx, y=dy, moving_time=0.08)
            time.sleep(0.1)
            x0, y0 = x, y

        print('Pen up')
        # Pen up
        self.bot.arm.set_ee_cartesian_trajectory(z=self.pen_height_diff, moving_time=0.3)
        time.sleep(0.1)

    def convert_to_canvas_coords(self, path):
        path = np.array(path)
        path[:, 0] = path[:, 0] / 1000
        path[:, 1] = path[:, 1] / 1000
        path[:, 0] = path[:, 0] + self.origin[0]
        path[:, 1] = path[:, 1] + self.origin[1]
        return path

    def draw_image(self, image_path, debug=False):
        print("Generating sketch paths...")
        resampled_paths = generate_sketch_paths(image_path, target_spacing=5.0, size=self.canvas_size, debug=debug)
        # resampled_paths = generate_sketch_paths(image_path, target_spacing=7.0, size=None, debug=debug)
        resampled_paths.sort(key=len, reverse=True)
        print(f'Canvas Bounds X: {self.canvas_bounds_x}')
        print(f'Canvas Bounds Y: {self.canvas_bounds_y}')

        # self.bot.arm.go_to_sleep_pose()
        self.bot.arm.go_to_home_pose(moving_time=2)
        time.sleep(1)
        
        # print("Checking canvas bounds...")
        # self.check_canvas_bounds()

        # print("Checking Pen Height Positions..")
        # self.check_pen_height_pos()

        # print(self.origin)
        print("Drawing image...")
        for i, path in enumerate(resampled_paths):
        # start_i = 8
        # for i in range(start_i, len(resampled_paths)):
        #     path = resampled_paths[i]
            print(f'Path {i+1}/{len(resampled_paths)}. Path length: {len(path)}')
            path = self.convert_to_canvas_coords(path)
            # plt.plot(path[:, 0], path[:, 1], 'r')
            # plt.show()
            self.draw_path(path)
            # break

        print("Drawing complete!")

        self.quit()