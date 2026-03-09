import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from utils import read_image, show_image


# BEGIN YOUR IMPORTS
from skimage.morphology import dilation, footprint_rectangle
from skimage.transform import rescale
from skimage.filters import gaussian
# END YOUR IMPORTS


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def find_edges(image):
    """
    Args:
        image (np.array): (grayscale) image of shape [H, W]
    Returns:
        edges (np.array): binary mask of shape [H, W]
    """
    # BEGIN YOUR CODE

    edges = cv2.Canny(image, 50, 150)
    
    return edges
    
    # END YOUR CODE


def highlight_edges(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        highlighted_edges (np.array): binary mask of shape [H, W]
    """
    # BEGIN YOUR CODE

    # highlited_edges = dilation(edges, footprint_rectangle((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    highlited_edges = cv2.dilate(edges, kernel, iterations=1)

    return highlited_edges
    
    # END YOUR CODE


def find_contours(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        contours (np.array, np.array, ...): tuple of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    """
    # BEGIN YOUR CODE

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours
    
    # END YOUR CODE


def get_max_contour(contours):
    """
    Args:
        contours (np.array, np.array, ...): tuple of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    Returns:
        max_contour (np.array): an array of points (vertices) of the contour with the maximum area of shape [N, 1, 2]
    """
    # BEGIN YOUR CODE

    max_contour = max(contours, key=cv2.contourArea)
    
    return max_contour
    
    # END YOUR CODE


def order_corners(corners):
    """
    Args:
        corners (np.array): an array of corner points (corners) of shape [4, 2]
    Returns:
        ordered_corners (np.array): an array of corner points in order [top left, top right, bottom right, bottom left]
    """
    # BEGIN YOUR CODE

    sum_points = corners.sum(axis=1)  # x + y of each point
    diff_points = corners[:, 0] - corners[:, 1]  # x - y of each point

    top_left = corners[np.argmin(sum_points)]
    top_right = corners[np.argmin(diff_points)]
    bottom_right = corners[np.argmax(sum_points)]
    bottom_left = corners[np.argmax(diff_points)]

    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])
    
    return ordered_corners
    
    # END YOUR CODE


def find_corners(contour, epsilon=0.02):
    """
    Args:
        contour (np.array): an array of points (vertices) of the contour of shape [N, 1, 2]
        epsilon (float): how accurate the contour approximation should be
    Returns:
        ordered_corners (np.array): an array of corner points (corners) of quadrilateral approximation of contour of shape [4, 2]
                                    in order [top left, top right, bottom right, bottom left]
    """
    # BEGIN YOUR CODE

    perimeter = cv2.arcLength(contour, True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon * perimeter, True)
    corners = approximated_contour.reshape(-1, 2)

    # to avoid errors
    if len(corners) != 4:
        np.append(corners, np.array([[0, 0],
                                     [0, 1],
                                     [1, 0],
                                     [1, 1]]), axis=0)
        corners = corners[:4]

    ordered_corners = order_corners(corners)

    return ordered_corners
    
    # END YOUR CODE


def rescale_image(image, scale=0.42):
    """
    Args:
        image (np.array): input image
        scale (float): scale factor
    Returns:
        rescaled_image (np.array): 8-bit (with range [0, 255]) rescaled image
    """
    # BEGIN YOUR CODE
    
    rescaled_image = rescale(image, scale, anti_aliasing=True, preserve_range=True).astype(np.uint8)
    
    return rescaled_image
    
    # END YOUR CODE


def gaussian_blur(image, sigma):
    """
    Args:
        image (np.array): input image
        sigma (float): standard deviation for Gaussian kernel
    Returns:
        blurred_image (np.array): 8-bit (with range [0, 255]) blurred image
    """
    # BEGIN YOUR CODE
    
    blurred_image = gaussian(image, sigma=sigma, preserve_range=True).astype(np.uint8)
    
    return blurred_image
    
    # END YOUR CODE


def distance(point1, point2):
    """
    Args:
        point1 (np.array): n-dimensional vector
        point2 (np.array): n-dimensional vector
    Returns:
        distance (float): Euclidean distance between point1 and point2
    """
    # BEGIN YOUR CODE

    distance = np.linalg.norm(point1 - point2)

    return distance
    
    # END YOUR CODE


def frontalize_image(image, ordered_corners):
    """
    Args:
        image (np.array): input image
        ordered_corners (np.array): corners in order [top left, top right, bottom right, bottom left]
    Returns:
        warped_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    # 4 source points
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    # BEGIN YOUR CODE

    length_top = distance(top_left, top_right)
    length_bottom = distance(bottom_left, bottom_right)
    length_left = distance(top_left, bottom_left)
    length_right = distance(top_right, bottom_right)

    # the side length of the Sudoku grid based on distances between corners
    side = int(max([length_top, length_bottom, length_left, length_right]))

    # what are the 4 target (destination) points?
    destination_points = np.array([[0, 0],
                                   [0, side],
                                   [side, side],
                                   [side, 0]], dtype=np.float32)

    # perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32),
                                                   destination_points)

    # image warped using the found perspective transformation matrix
    warped_image = cv2.warpPerspective(image, transform_matrix, (side, side))

    assert warped_image.shape[0] == warped_image.shape[1], "height and width of the warped image must be equal"

    return warped_image

    # END YOUR CODE


def show_frontalized_images(image_paths, pipeline, figsize=(16, 12)):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for i, image_path in enumerate(tqdm(image_paths)):
        axis = axes[i // ncols][i % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image, _ = pipeline(sudoku_image)

        show_image(frontalized_image, axis=axis, as_gray=True)
