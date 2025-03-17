import os
import cv2
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from itertools import permutations
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

directory_path = r'C:\Users\13601\Desktop'
image_path = os.path.join(directory_path, 'my_picture.jpg')

# FUNCTIONS DEFINITION --------------------------------------------------------------------
def rgb_to_binary(image_path, blur_kernel=(5, 5), canny_thresholds=(50, 150)):
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

    # 1. Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 3. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 4. Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])

    # 5. yield binary image
    binary_image = (edges > 0).astype(np.uint8)

    return binary_image

def tsp_nearest_neighbor(points):
    """Solve TSP using a nearest neighbor heuristic."""
    if len(points) < 2:
        return points

    # Convert NumPy arrays to tuples for hashing
    points = [tuple(p) for p in points]

    path = [points[0]] # The path starts with the first point.
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
        return euclidean(point, start)

    x0, y0 = point
    x1, y1 = start
    x2, y2 = end

    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return num / den

def douglas_peucker(points, epsilon):
    """Simplify the path using the Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points  # No need to simplify if only two points

    start, end = points[0], points[-1]
    max_dist = 0
    index = 0

    # Find the point with the maximum distance from the line (start-end)
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            index = i

    # If max distance is greater than epsilon, keep that point and recursively simplify
    if max_dist > epsilon:
        left = douglas_peucker(points[:index+1], epsilon)
        right = douglas_peucker(points[index:], epsilon)
        return left[:-1] + right  # Merge both parts, removing duplicate middle point

    else:
        return [start, end]  # Remove all points except endpoints if deviation is small

# FUNCTIONS DEFINITION ENDS----------------------------------------------------------------

"""
Obtain binary result of the image. 
"""
binary_result = rgb_to_binary(image_path)

"""
From the binary image, extract edge points. 
"""
# Extract contours from the binary image.
contours, _ = cv2.findContours(binary_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert into a list of (x, y) edge points.
edge_points = [tuple(pt[0]) for contour in contours for pt in contour]

"""
From the collection of edge points, randomly select a set of sample points. 
"""
# set the number of sampled points.
num_samples = min(3000, len(edge_points)) # You can replace '3000' with other values, as you see fit.
sampled_nodes = random.sample(edge_points, num_samples)

"""
From the collection of sampled points, detect clusters using DBSCAN
"""
# Convert sampled points to NumPy array
node_array = np.array(sampled_nodes)

# Apply DBSCAN clustering
eps = 5  # Max distance between points in a cluster
min_samples = 2  # Minimum number of points to form a cluster
db = DBSCAN(eps=eps, min_samples=min_samples).fit(node_array)

# Assign cluster labels
cluster_labels = db.labels_  # -1 means noise (not part of any cluster)

# Organize nodes by cluster
clusters = {}
for point, label in zip(sampled_nodes, cluster_labels):
    if label != -1:  # Ignore noise
        clusters.setdefault(label, []).append(point)

"""
Now, note that a collection of cluster is optained, i.e. clusters
Within each cluster, find a path to traverse all points. (i.e. draw inside each cluster)
"""
# Compute TSP paths for each cluster
cluster_paths = {}
for label, cluster_points in clusters.items():
    cluster_paths[label] = tsp_nearest_neighbor(cluster_points)

"""
Path smoothing to be performed. 
"""
# Apply the Douglas-Peucker algorithm to each cluster path
epsilon = 1.5  # Adjust epsilon for more/less simplification
cluster_paths = {label: douglas_peucker(path, epsilon) for label, path in cluster_paths.items()}

"""
Now, find a path to traverse all clusters. 
"""
# Prepare data. 
cluster_centroids = {label: np.mean(points, axis=0) for label, points in clusters.items()} # Compute centroids of each cluster
centroids = np.array(list(cluster_centroids.values()))  # Cluster centroids
labels = list(cluster_centroids.keys())  # Cluster labels

# Construct a Graph
G = nx.Graph()

# Add nodes into the Graph (each centroid is a node)
for i, label in enumerate(labels):
    G.add_node(i, pos=centroids[i])  # Assign centroid position

# Compute pairwise Euclidean distances
dist_matrix = distance_matrix(centroids, centroids)

# Add weighted edges
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        weight = dist_matrix[i, j]
        G.add_edge(i, j, weight=weight)

# Solve TSP using NetworkX approximation
tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)

# Plot intra-cluster paths (red)
for label, path in cluster_paths.items():
    x_p, y_p = zip(*path)
    plt.plot(x_p, y_p, 'red', linewidth=0.5, label=f"Cluster {label}")

# Plot TSP path between clusters (blue)
for i in range(len(tsp_path) - 1):
    start, end = centroids[tsp_path[i]], centroids[tsp_path[i + 1]]
    plt.plot([start[0], end[0]], [start[1], end[1]], 'k--', linewidth=0.3)

plt.title("Approximate TSP Path Between Clusters")
plt.show()

"""
Return information for downstream processing.
"""
var1 = cluster_paths
var2 = tsp_path
print("var1: information about how to draw each individual cluster.")
print("var2: information about how to traverse between clusters, thus drawing all clusters (i.e. the entire image)")

# Last editted, March 16, 2025. 