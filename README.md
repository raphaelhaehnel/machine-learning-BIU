# Machine Learning Training

This repositories contains implementations of the basics machine learning algorithms.
The exercices has been done as a part of the studies in Bar-Ilan University.
The instructions has been written by Alan Joseph Bekker

## Ex1
This exercise implements linear regression by generating a random feature matrix $`X`$, defining true parameters $`\beta`$, adding noise, computing $`Y = X\beta + \varepsilon`$, and estimating $`\beta`$ using the normal equation. The effect of increasing noise variance on parameter estimation is also analyzed.

## Ex2
This exercise trains a binary logistic regression model using gradient ascent to distinguish between digits 1 and 2. The cost function is optimized iteratively, visualized, and the success rate is evaluated.

## Ex3
This exercise trains a multi-class logistic regression model using gradient ascent to classify all digits in the MNIST dataset. The cost function is optimized iteratively, the data is visualized, and the success rate is evaluated.

## Ex4
This exercise applies the K-means clustering algorithm to the MNIST dataset to group data into 10 clusters. The cost function is monitored for convergence, and the success rate is evaluated.

## Ex5
This exercise applies the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (GMM) to cluster 2D data generated from three Gaussians. K-means and GMM are compared based on their clustering accuracy against the true labels.

![gmm moving](https://github.com/user-attachments/assets/6733cc70-fc2f-4cc8-9f6f-a99eac221eb9)

## Ex6
This exercise implements a nearest-neighbor face recognition algorithm using Eigenfaces and PCA. The model is trained on facial images, projects test images into a lower-dimensional space, and classifies them based on Euclidean distance to training images, analyzing classification performance as a function of K.
