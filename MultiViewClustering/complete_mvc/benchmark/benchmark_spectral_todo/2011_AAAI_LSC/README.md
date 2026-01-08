# Large Scale Spectral Clustering with Landmark-Based Representation
The purpose of this code is mainly educational. It is the implementation of unsupervised learning technique called: Large Scale Spectral Clustering with Landmark-Based Representation

## LSC Algorithm Explanation
LSC Algorithm Idea: Use a few "landmarks" to approximate the relationship between all data points, reducing computational complexity.

### Algorithm Flow (4 Steps):
1. Select Landmarks
    - Select p representative points from the data (p << n, n is the number of samples)
    - Method: Random selection or K-means cluster centers
2. Use landmarks to represent all data points
    - For each data point, calculate its similarity to all landmarks (Gaussian kernel)
    - Only keep the weights of the r most similar landmarks for each point (sparse coding)
    - Get the coding matrix Z: (p, n), representing which landmarks each point is a linear combination of
3. Dimensionality Reduction (SVD)
    - Perform SVD on the coding matrix, take the top k principal components
    - Get the low-dimensional embedding: (n, k), k is the number of clusters
4. Clustering
    - Perform K-means in the low-dimensional embedding space

### Why is it effective?
- Traditional spectral clustering needs to calculate the similarity matrix between all pairs of points (n, n), complexity O(n²)
- LSC only calculates the similarity between each point and landmarks, matrix (p, n), p << n, complexity O(pn)
- Approximate the structure relationship between data points through sparse coding of landmarks
- Summary: Use landmarks as a "bridge" to reduce the similarity calculation of large-scale data to a small-scale problem, and then cluster in the low-dimensional space.

## Python Version
This repository contains a Python implementation converted from the original Julia code.

### Requirements
```bash
pip install numpy scipy scikit-learn
```

### Original Julia Version
The original Julia code is also available in `LSC.jl` and `Evaluation.jl` files.

Please visit http://int8.io/large-scale-spectral-clustering-with-landmark-based-representation for details (+ to see some experiments)
