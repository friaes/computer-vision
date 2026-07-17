# Computer Vision — Course Assignments

Assignments from the **Computer Vision** course at the **University of Bern**, taken as part of my Master's in Computer Science. The course covered image formation, image processing, feature detection, segmentation, multiple-view geometry, 3D reconstruction, motion, and object recognition.

This repository contains three assignments, each a self-contained Python project built on the scientific-Python / OpenCV stack:

| # | Assignment | Topic | Key techniques |
|---|-----------|-------|----------------|
| 1 | [Sudoku recognition & solving](#assignment-1--sudoku-recognition--solving) | Classic image-processing pipeline | Edge/contour detection, perspective correction, template matching, backtracking solver |
| 2 | [Image inpainting](#assignment-2--image-inpainting) | Variational reconstruction | Energy minimisation, sparse Hessian, gradient descent / Gauss–Seidel |
| 3 | [Multiple-view geometry](#assignment-3--multiple-view-geometry--3d-reconstruction) | Two-view 3D reconstruction | 8-point algorithm, epipolar geometry, RANSAC, triangulation |

## Assignment 1 — Sudoku recognition & solving

**Folder:** `assignment/`

An end-to-end pipeline that takes a photo of a Sudoku puzzle, reads the digits off the grid, and solves it. It's a tour through the classic image-processing toolkit, wired together with a small composable `Pipeline` class that can visualise every intermediate stage.

The stages:

1. **Edge detection** — Canny edges, then morphological dilation to close gaps.
2. **Contour & corner detection** — find the largest external contour (the puzzle boundary), approximate it to a quadrilateral (`approxPolyDP`), and order its four corners.
3. **Frontalization** — compute a perspective transform from the detected corners (`getPerspectiveTransform`) and warp the puzzle to a fronto-parallel square (`warpPerspective`), removing the camera's viewing angle.
4. **Cell extraction & binarization** — split the frontalized grid into 81 cells and adaptively threshold each one; detect empty cells by their white-pixel ratio.
5. **Digit recognition** — classify each non-empty cell by **normalized cross-correlation template matching** (`match_template`) against a bank of digit templates (`templates/1`…`templates/9`), taking the best-correlating digit above a confidence threshold.
6. **Solving** — feed the recognised grid into a constraint-propagation + backtracking Sudoku solver.

**Files:** `frontalization.py` (steps 1–3), `recognition.py` (steps 4–5), `sudoku_solver.py` (step 6), `pipeline.py` (the stage orchestrator + visualisation), `template.py` / `create_templates.py` (digit template bank), `utils.py`, plus `sudoku_solver.ipynb` for running and visualising the whole thing. Sample puzzles live in `sudoku_puzzles/` and cached results in `frontalized_images/`.

<p float="left">
  <img src="images/image copy.png" width="474" />
  <img src="images/image.png" width="400" /> 
</p>


## Assignment 2 — Image inpainting

**Folder:** `assignment2/`

Reconstruct the missing region of a masked image by treating inpainting as an **energy-minimisation** problem. Given a masked image `g` and a binary mask `omega` (1 where pixels are known), we recover the full image `u` by minimising a data term (stay faithful to the known pixels) plus a smoothness regulariser weighted by `λ`.

The core deliverable is `hessian_matrix.py`, which assembles the **Hessian of that energy functional** as a sparse matrix. It builds, for every pixel, the coefficients linking it to its neighbourhood (a discrete Laplacian-style stencil) with careful handling of the image boundaries, and returns a `scipy.sparse` operator over the whole `M×N` image. The notebook then solves the system two ways:

- **Gradient Descent (GD)** on the energy, re-clamping known pixels each step.
- **Linearised Gauss–Seidel (LGS)** iterative solves.

**Files:** `hessian_matrix.py` (the sparse energy Hessian), `Assignment2.ipynb` (the GD/LGS solvers, experiments, and results), plus the assignment brief and worked exercises as PDFs.

## Assignment 3 — Multiple-view geometry & 3D reconstruction

**Folder:** `assingnment3/`

Two-view geometry on the classic **Merton College** (Oxford VGG) dataset: recover the epipolar geometry between two images and reconstruct 3D structure from 2D correspondences.

What it implements (in `utils.py`):

- **Normalised 8-point algorithm** (`eight_points_algorithm`) — estimate the fundamental matrix from point correspondences, with coordinate normalisation, an SVD solve, and rank-2 enforcement.
- **Epipoles & epipolar lines** — compute the epipole as the null space of `F` and plot the epipolar lines that every correspondence must lie on.
- **RANSAC** (`ransac`) — robustly estimate `F` in the presence of outlier matches by scoring candidates on their epipolar-distance inliers.
- **Essential-matrix decomposition** (`decompose_essential_matrix`) — factor the essential matrix into the relative camera rotation and translation.
- **Triangulation** (`infer_3d`) — reconstruct 3D points from the two camera projections, resolving the correct solution by requiring points to lie in front of both cameras.

**Files:** `utils.py` (all of the above), `data.py` (dataset loading), `Assignment3_part1.ipynb` / `Assignment3_part2.ipynb` (the experiments), and the `merton_college/` dataset with its 2D/3D points, lines, corners, and camera matrices.

## Requirements

Python 3 with the scientific-vision stack: `numpy`, `scipy`, `opencv-python`, `scikit-image`, `matplotlib`, `pillow`, `tqdm`, and Jupyter to run the notebooks.

```bash
pip install numpy scipy opencv-python scikit-image matplotlib pillow tqdm jupyter
```

## Usage

Each assignment is driven from its notebook:

```bash
# Assignment 1
jupyter notebook assignment/sudoku_solver.ipynb

# Assignment 2
jupyter notebook assignment2/Assignment2.ipynb

# Assignment 3
jupyter notebook assingnment3/Assignment3_part1.ipynb
jupyter notebook assingnment3/Assignment3_part2.ipynb
```

The `.py` files hold the reusable implementations that the notebooks import and call.

## Repository layout

```
.
├── assignment/       # Assignment 1 — Sudoku recognition & solving
│   ├── frontalization.py   recognition.py   sudoku_solver.py
│   ├── pipeline.py   template.py   create_templates.py   utils.py
│   ├── templates/ 1..9      sudoku_puzzles/   frontalized_images/
│   └── sudoku_solver.ipynb
├── assignment2/      # Assignment 2 — Image inpainting
│   ├── hessian_matrix.py    Assignment2.ipynb    *.pdf
├── assingnment3/     # Assignment 3 — Multiple-view geometry
│   ├── utils.py   data.py
│   ├── Assignment3_part1.ipynb   Assignment3_part2.ipynb
│   └── merton_college/  MatchedPoints/
└── README.md
```
