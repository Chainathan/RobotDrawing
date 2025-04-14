import cv2
import numpy as np
from interbotix_common_modules.common_robot.robot import robot_startup, robot_shutdown
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import time

# CONFIGURATION
image_path = "/home/pablo/Desktop/circle.jpg"  # Path to input image
x_offset = 0.20  # bottom-left corner x (m)
y_offset = -0.05  # bottom-left corner y (m)
drawing_area = 0.10  # 10 cm square (meters)
z_draw = 0.0
z_lift = 0.05
execute = True  # Set to True to actually move the robot

# LOAD AND PROCESS IMAGE
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 9, 75, 75)
binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 11, 2)
edges = cv2.Canny(filtered, 50, 150)
combined = cv2.bitwise_or(binary, edges)

# SKELETONIZATION
from skimage.morphology import skeletonize, label
skel = skeletonize(combined > 0).astype(np.uint8)
labels = label(skel)

# EXTRACT PATHS
def get_neighbors(y, x, img):
    h, w = img.shape
    return [(j, i) for j in range(y-1, y+2) for i in range(x-1, x+2)
            if (0 <= j < h and 0 <= i < w and img[j, i] and (j, i) != (y, x))]

def extract_paths(labeled_img):
    paths = []
    for label_id in np.unique(labeled_img):
        if label_id == 0:
            continue
        mask = (labeled_img == label_id).astype(np.uint8)
        coords = np.column_stack(np.nonzero(mask))
        endpoints = [tuple(pt) for pt in coords if len(get_neighbors(*pt, mask)) == 1]
        if not endpoints and len(coords) > 0:
            endpoints = [tuple(coords[0])]

        visited = set()
        for ep in endpoints:
            if ep in visited:
                continue
            path = []
            current = ep
            while current and current not in visited:
                path.append((current[1], current[0]))  # x, y
                visited.add(current)
                neighbors = [n for n in get_neighbors(*current, mask) if n not in visited]
                current = neighbors[0] if neighbors else None
            if len(path) > 1:
                paths.append(np.array(path))
    return paths

paths = extract_paths(labels)

# SMOOTH AND SCALE PATHS
h, w = skel.shape
scaled_paths = []
for path in paths:
    if len(path) < 2:
        continue
    # Smooth using moving average
    kernel = 5
    smoothed = np.convolve(path[:,0], np.ones(kernel)/kernel, mode='valid'), \
               np.convolve(path[:,1], np.ones(kernel)/kernel, mode='valid')
    smoothed = np.vstack(smoothed).T

    # Scale and translate to robot coords
    scaled = np.zeros((smoothed.shape[0], 3))
    scaled[:, 0] = x_offset + (smoothed[:, 0] / w) * drawing_area
    scaled[:, 1] = y_offset + (smoothed[:, 1] / h) * drawing_area
    scaled[:, 2] = z_draw
    scaled_paths.append(scaled)

# EXECUTE
if execute:
    bot = InterbotixManipulatorXS("wx250s", "arm", "gripper")
    robot_startup()
    bot.arm.go_to_home_pose()
    bot.arm.set_ee_pose_components(pitch=-1.57)

    for path in scaled_paths:
        start = path[0]
        bot.arm.set_ee_pose_components(x=start[0], y=start[1], z=z_lift)
        bot.arm.set_ee_cartesian_trajectory(z=-z_lift)
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            bot.arm.set_ee_cartesian_trajectory(x=dx, y=dy)
        bot.arm.set_ee_cartesian_trajectory(z=z_lift)

    bot.arm.go_to_sleep_pose()
    robot_shutdown()
