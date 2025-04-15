import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance, KDTree
from skimage.morphology import skeletonize, remove_small_objects
from scipy.interpolate import splprep, splev, interp1d
from PIL import Image
# import trimesh
from collections import deque


def rgb_to_binary(image, blur_kernel=(5, 5), canny_thresholds=(50, 150)):
    """
    Converts an RGB image into a binary image, retaining all essential features.
    0s represent background,
    1s represent edges.

    Parameters:
        image_path (str): Path to the input RGB image.
        blur_kernel (tuple): Kernel size for Gaussian blur (default: (5,5)).
        canny_thresholds (tuple): Min and max thresholds for Canny edge detection (default: (50,150)).

    Returns:
        binary_image (numpy.ndarray): 2D binary image with values 0 (background) and 1 (edges).
    """
    # 1. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)

    # 2. Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])

    # 3. yield binary image
    binary_image = (edges > 0).astype(np.uint8)

    return binary_image

def create_graph(binary_image):
    G = nx.Graph()
    
    # Add nodes to the graph
    rows, cols = binary_image.shape
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:
                G.add_node((j, i))
    
    # Add edges to the graph
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:
                # Connect with the right pixel
                if j < cols - 1 and binary_image[i, j + 1] == 1:
                    G.add_edge((j, i), (j + 1, i))
                # Connect with the pixel below
                if i < rows - 1 and binary_image[i + 1, j] == 1:
                    G.add_edge((j, i), (j, i + 1))
                # Connect with the bottom-right pixel
                if i < rows - 1 and j < cols - 1 and binary_image[i + 1, j + 1] == 1:
                    G.add_edge((j, i), (j + 1, i + 1))
                # Connect with the bottom-left pixel
                if i < rows - 1 and j > 0 and binary_image[i + 1, j - 1] == 1:
                    G.add_edge((j, i), (j - 1, i + 1))
    return G

def show_images(images, titles=None):
    if titles is None:
        titles = ['Image %d' % i for i in range(1, len(images) + 1)]
    n = len(images)
    if n == 1:
        plt.figure()
        plt.imshow(images[0], cmap='gray')
        plt.axis('off')
        plt.title(titles[0])
    else:
        fig, axes = plt.subplots(1, n, figsize=(15, 15))
        for i in range(n):
            axes[i].imshow(images[i], cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(titles[i])
    plt.show()

def extract_branches(graph):
    branches = []
    visited = set()

    # Find all junction nodes (nodes with degree > 2)
    junctions = {node for node in graph.nodes() if graph.degree(node) > 2}

    # Traverse the graph to extract linear paths (branches)
    for node in graph.nodes():
        if node in visited:
            continue

        if graph.degree(node) == 1 or node in junctions:  # Start at leaf or junction
            for neighbor in graph.neighbors(node):
                if (node, neighbor) not in visited and (neighbor, node) not in visited:
                    branch = []
                    queue = deque([(node, neighbor)])  # Start BFS from the branch start

                    while queue:
                        prev, curr = queue.popleft()
                        branch.append(curr)
                        visited.add((prev, curr))
                        visited.add((curr, prev))

                        # If we reach a junction or endpoint, stop
                        if graph.degree(curr) > 2 or graph.degree(curr) == 1:
                            break

                        # Continue to the next neighbor
                        for next_node in graph.neighbors(curr):
                            if (curr, next_node) not in visited:
                                queue.append((curr, next_node))
                                break  # Move in one direction only

                    branches.append([node] + branch)

    return branches

def resample_path(path, target_spacing=2.0):
    """
    Resample a path to have points spaced at a fixed distance.

    :param path: np.array of shape (N, 2), original list of (x, y) coordinates
    :param target_spacing: float, desired distance between consecutive points
    :return: np.array of shape (M, 2), resampled path
    """
    # Compute cumulative distances along the path
    distances = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Insert 0 at start

    # Create interpolation functions for x and y
    interp_x = interp1d(distances, path[:, 0], kind='linear')
    interp_y = interp1d(distances, path[:, 1], kind='linear')

    # Generate new sampled distances with equal spacing
    new_distances = np.arange(0, distances[-1], target_spacing)

    # Compute new points
    new_x = interp_x(new_distances)
    new_y = interp_y(new_distances)

    return np.column_stack((new_x, new_y))

def generate_sketch_paths(image_path, target_spacing=5.0, size=None, debug=False):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if size is not None:
        print('size:', size)
        size = tuple(int(x * 1000) for x in size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    binary_result = rgb_to_binary(image)
    skeleton = skeletonize(binary_result)

    # Remove isolated pixels
    skeleton = remove_small_objects(skeleton, connectivity=2, min_size=7)

    # Convert points into a graph
    G = create_graph(skeleton)

    # Extract connected components
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    resampled_paths = []

    # Extract branches from each component and resample them
    for idx, component in enumerate(components):
        branches = extract_branches(component)  # Extract branches from component
        for branch in branches:
            resampled_points = resample_path(np.array(branch), target_spacing)
            resampled_paths.append(resampled_points)

    resampled_paths = [path for path in resampled_paths if len(path) > 1]

    if debug:
        # Number of points
        num_points = sum(len(path) for path in resampled_paths)
        print("Number of points in original Image:", np.sum(binary_result))
        print("Number of points after skeletonization:", len(G.nodes()))
        print("Number of points after resampling:", num_points)

        # Plot the resampled paths
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton, cmap='gray')
        for path in resampled_paths:
            # plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=1)
            plt.scatter(path[:, 0], path[:, 1], s=1, c='r')
        # plt.gca().invert_yaxis()
        plt.axis('equal')
        plt.title("Resampled Paths")
        plt.show()

    return resampled_paths