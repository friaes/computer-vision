from pipeline import Pipeline

# BEGIN YOUR IMPORTS
from frontalization import gaussian_blur, find_edges, highlight_edges, find_contours, get_max_contour, find_corners, frontalize_image
from recognition import resize_image, get_sudoku_cells
from const import SUDOKU_SIZE
# END YOUR IMPORTS

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example

CELL_COORDINATES = {"image_0.jpg": {1: (0, 0),
                                    2: (1, 1)},
                    "image_2.jpg": {1: (2, 3),
                                    3: [(2, 1), (0, 4)],
                                    9: (5, 6)}}
"""
# chose these images because of different number fonts
CELL_COORDINATES = {
    "image_0.jpg": {
        1: (6, 4),
        2: (2, 8),
        3: (1, 3),
        4: (1, 4),
        5: (4, 0),
        6: (4, 4),
        7: (2, 4),
        8: (0, 5),
        9: (2, 0)
    },
    "image_5.jpg": {
        1: (2, 5),    
        2: (5, 5), 
        3: (4, 5), 
        4: (0, 0),
        5: (1, 1),
        6: (0, 8),
        7: (0, 5),
        8: (4, 1),
        9: (3, 3)
    }
}

# END YOUR CODE


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def get_template_pipeline():
    # BEGIN YOUR CODE

    pipeline = Pipeline(
        functions=[
            gaussian_blur, 
            find_edges, 
            highlight_edges, 
            find_contours, 
            get_max_contour, 
            find_corners, 
            frontalize_image,
            resize_image, 
            get_sudoku_cells
        ],
        parameters={
            "gaussian_blur": {"sigma": 0.5},
            "find_corners": {"epsilon": 0.02},
            "resize_image": {"size": SUDOKU_SIZE},
            "get_sudoku_cells": {
                "crop_factor": 0.62, 
                "binarization_kwargs": {
                    'block_size': 17, 
                    'c_constant': 18
                }
            }
        }
    )
    
    return pipeline

    # END YOUR CODE
