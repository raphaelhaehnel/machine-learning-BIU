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

- The following figure shows the decomposition of a face with 6 different components, and the mean of all the samples
![show_PCA](https://github.com/user-attachments/assets/e87b1a6a-12c3-495d-8855-459b21c427a3)

- The following figure shows the reconstruction of the image, by using more and more components
![reconstruct_image](https://github.com/user-attachments/assets/09d6a7ee-af00-4883-95ff-62631716cc0c)

## Bonus
Animation of the K-means algorithm, in 2d and 3d

![2d-animation](https://user-images.githubusercontent.com/69756617/207129209-124e0738-e074-42bf-9a15-248583a1051b.gif)
![3d-animation](https://user-images.githubusercontent.com/69756617/207129236-741b5c15-6f14-404c-a721-9ee6f93cc319.gif)

