# Fashion MNIST Classification with PCA and Random Forest

This project applies Principal Component Analysis (PCA) and Random Forest Classification to the **Fashion MNIST** dataset. The goal is to demonstrate the effect of PCA on model performance, particularly how dimensionality reduction impacts both accuracy and training time.

## Project Overview

- **Dataset**: Fashion MNIST, a dataset of 60,000 28x28 grayscale images of 10 fashion categories, and 10,000 test images.
- **PCA**: PCA is applied for dimensionality reduction. By transforming the dataset into a lower-dimensional space, PCA aims to retain the maximum variance in the data with fewer features.
- **Model**: Random Forest Classifier is used for training. Random Forest is an ensemble method that combines multiple decision trees to make predictions, offering high accuracy and generalization.

## Libraries Used

- `TensorFlow` for loading the Fashion MNIST dataset.
- `matplotlib` for visualizations.
- `scikit-learn` for PCA, Random Forest, and performance metrics.
- `numpy` for numerical operations.

## PCA and Random Forest Overview

### Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that transforms a large set of variables into a smaller one while retaining most of the information. It is useful for:
- Reducing computational cost.
- Handling multicollinearity in data.
- Visualizing high-dimensional data.

### Random Forest Classifier

Random Forest is an ensemble learning method that uses multiple decision trees to improve accuracy and prevent overfitting. It is:
- Robust and works well with high-dimensional data.
- Used for both classification and regression tasks.
- Less prone to overfitting compared to individual decision trees.

## Steps and Visualizations

### 1. **Dataset Visualization**

The dataset consists of grayscale images representing fashion items, each labeled with one of 10 categories. Below, we visualize a 5x5 grid of images from the training set to provide a sense of the dataset's diversity.

![Data Visualization](https://github.com/user-attachments/assets/fe507975-3aed-4711-8f5d-528500a0d9ef)

### 2. **Explained Variance Ratio by Number of Components (PCA)**

We use PCA to reduce the dimensionality of the dataset. The plot below shows how the explained variance ratio increases as the number of components increases. This helps in determining how many components are necessary to capture a sufficient amount of information from the original dataset.

![Explained Variance Plot](https://github.com/user-attachments/assets/15ef1532-df5e-4ead-9b65-5f14498d5fb5)

### 3. **Accuracy and Time vs. Number of Components**

The following image shows two plots:
- **Accuracy vs. Number of Components**: This plot shows the model's accuracy with varying numbers of principal components.
- **Time Taken vs. Number of Components**: This plot demonstrates the computational cost (in terms of time) for training the model as the number of components changes.

![Accuracy and Time Comparison](https://github.com/user-attachments/assets/1c36e05c-015f-4f11-8110-fe9ae3fc232a)

### 4. **Comparison of Models with and without PCA**

We compare the performance of the Random Forest model trained with and without PCA. Below is a side-by-side bar chart showing:
- **Accuracy Comparison**: Accuracy of the Random Forest model with and without PCA.
- **Time Comparison**: Time taken for training and prediction for both models.

![Model Comparison](https://github.com/user-attachments/assets/81e26483-d381-4c52-aa30-263c3e68877d)

## Results Summary

- **Accuracy with PCA**: 0.8529
- **Accuracy without PCA**: 0.8760
- **Time Taken with PCA**: 109.6082 seconds
- **Time Taken without PCA**: 146.5678 seconds

From the comparison, we can observe that while the model with PCA has a slightly lower accuracy, it benefits from significantly reduced computation time.

## How to Run the Code

1. Clone the repository.
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter notebook to see the results step-by-step.

## Conclusion

This project demonstrates the effect of applying PCA to a Random Forest classifier on the Fashion MNIST dataset. PCA helps reduce the dimensionality of the data, offering faster training times at the cost of a slight reduction in accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

