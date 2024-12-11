# CSE803-Project

## Unsupervised Intrinsic Image Decomposition with Implicit Neural Representations

### Group Members: Fizza Rubab and Aarush Mathur

This project explores the application of Neural Fields (implicit neural representations) for intrinsic image decomposition, aiming to separate an image into its reflectance (albedo) and shading components. Using a multi-layer perceptron (MLP) with a coordinate-based representation, this method models intrinsic components as continuous functions over spatial coordinates, enabling resolution-independent predictions. The project leverages various priors like reflectance sparsity, chromaticity, and segmentation-based guidance to improve decomposition quality in a self-supervised manner.

1. `decompose.py` file implements the naive and prior method given an input image.
2. `decompose_segmentation.py` file implements the segmentation based guidance given an input image and segmentation map.
3. `quantitative_results.ipynb` file contains the error evaluations.
4. `segmentation.ipynb` file contains code to generate average color reflectance estimate.
