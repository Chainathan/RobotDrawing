import os
import cv2
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time
from path_planning_utils import generate_sketch_paths

class RobotClass:
    def __init__(self):
        '''
        The units are in millimeters. It will be converted to meters before feeding to the robot.
        '''
        grid_size = 250 / 1000
        # self.canvas_bounds_x = (grid_size/2, 2 * grid_size + grid_size/4)
        # self.canvas_bounds_y = (-grid_size, grid_size)

        # square large
        # side = 2 * grid_size + grid_size/4 - grid_size/2
        # self.canvas_bounds_x = (grid_size/2, 2 * grid_size + grid_size/4)
        # self.canvas_bounds_y = (-side/2, side/2)

        # square small
        side = grid_size
        self.canvas_bounds_x = (grid_size/2, grid_size/2 + side)
        self.canvas_bounds_y = (-side/2, side/2)

        self.canvas_size = (self.canvas_bounds_x[1] - self.canvas_bounds_x[0], self.canvas_bounds_y[1] - self.canvas_bounds_y[0])
        self.origin = (self.canvas_bounds_y[0], self.canvas_bounds_x[0])
        # self.origin = (self.canvas_bounds_x[0], self.canvas_bounds_y[0])
        self.pen_up_height = 110 / 1000
        self.pen_down_height = 100 / 1000
        self.pen_height_diff = self.pen_up_height - self.pen_down_height

        self.canvas_center = (self.canvas_bounds_x[0] + self.canvas_size[0]/2, self.canvas_bounds_y[0] + self.canvas_size[1]/2)

        # Initialize the robot interface
        self.bot = InterbotixManipulatorXS(
            robot_model='wx250s',
            group_name='arm',
            gripper_name='gripper',
        )

        robot_startup()

    def quit(self):
        self.bot.arm.go_to_home_pose()
        self.bot.arm.go_to_sleep_pose()
        robot_shutdown()

    def check_canvas_bounds(self):
        # Move the robot to the canvas bounds
        self.bot.arm.set_ee_pose_components(x=self.canvas_center[0], y=self.canvas_center[1], z=self.pen_up_height)
        bounding_points = [(self.canvas_bounds_x[0], self.canvas_bounds_y[0]), 
                        (self.canvas_bounds_x[1], self.canvas_bounds_y[0]), 
                        (self.canvas_bounds_x[1], self.canvas_bounds_y[1]), 
                        (self.canvas_bounds_x[0], self.canvas_bounds_y[1])]
        x0, y0 = self.canvas_center[0], self.canvas_center[1]
        for x, y in bounding_points:
            dx, dy = x - x0, y - y0
            self.bot.arm.set_ee_cartesian_trajectory(dx, dy, 0)
            x0, y0 = x, y

        dx, dy = self.canvas_center[0] - x0, self.canvas_center[1] - y0
        self.bot.arm.set_ee_cartesian_trajectory(dx, dy, 0)
        
    def draw_path(self, path):
        # Start at the first point
        print('First point:', path[0])
        x0, y0 = path[0]
        self.bot.arm.set_ee_pose_components(x=x0, y=y0, z=self.pen_up_height, moving_time=0.08)
        time.sleep(1)

        print('Pen down')
        # Pen down
        self.bot.arm.set_ee_cartesian_trajectory(z=-self.pen_height_diff, moving_time=0.08)
        time.sleep(0.1)

        print('Drawing path...')
        # Draw the path
        for x, y in path[1:]:
            dx, dy = x - x0, y - y0
            self.bot.arm.set_ee_cartesian_trajectory(x=dx, y=dy, moving_time=0.08)
            time.sleep(0.1)
            x0, y0 = x, y

        print('Pen up')
        # Pen up
        self.bot.arm.set_ee_cartesian_trajectory(z=self.pen_height_diff, moving_time=0.08)
        time.sleep(0.1)

    def convert_to_canvas_coords(self, path):
        path = np.array(path)
        path[:, 0] = path[:, 0] / 1000
        path[:, 1] = path[:, 1] / 1000
        path[:, 0] = path[:, 0] - self.origin[0]
        path[:, 1] = path[:, 1] - self.origin[1]
        return path

    def draw_image(self, image_path, debug=False):
        print("Generating sketch paths...")
        resampled_paths = generate_sketch_paths(image_path, target_spacing=5.0, size=self.canvas_size, debug=debug)
        print(f'Canvas Bounds X: {self.canvas_bounds_x}')
        print(f'Canvas Bounds Y: {self.canvas_bounds_y}')

        # self.bot.arm.go_to_sleep_pose()
        self.bot.arm.go_to_home_pose()
        
        print("Checking canvas bounds...")
        # self.check_canvas_bounds()

        print(self.origin)
        print("Drawing image...")
        for path in resampled_paths:
            path = self.convert_to_canvas_coords(path)
            print(path)
            plt.plot(path[:, 0], path[:, 1], 'r')
            plt.pause(1)
            plt.draw()
            self.draw_path(path)
            # break

        print("Drawing complete!")

        self.quit()