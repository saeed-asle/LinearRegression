# Linear Regression Optimization Techniques
Authored by saeed asle

# Description
This project explores various optimization techniques for linear regression, including gradient descent, mini-batch gradient descent, and momentum.
The dataset used in this project is related to cancer data, where the goal is to predict a target variable 'y' based on input features 'x0' to 'x8'.

# steps:
* Data Preprocessing: Loads the dataset, handles missing values, and standardizes the data to have zero mean and unit variance.
* Gradient Descent: Implements the gradient descent algorithm to minimize the cost function and find the optimal values for the model parameters.
* Mini-Batch Gradient Descent: Implements mini-batch gradient descent to update the model parameters using batches of data, which can lead to faster convergence.
* Momentum: Implements the momentum optimization technique to accelerate convergence by adding a momentum term to the parameter update.
* Evaluation: Evaluates the performance of each optimization technique in terms of convergence speed and final cost.
# Features
* Data Preprocessing: Handles missing values and standardizes the data.
* Gradient Descent: Minimizes the cost function to find optimal model parameters.
* Mini-Batch Gradient Descent: Updates model parameters using mini-batches of data.
* Momentum: Accelerates convergence by adding a momentum term to the parameter update.
* Evaluation Metrics: Computes the cost function and tracks convergence over time.
# Dependencies
* numpy: For numerical operations.
* pandas: For data manipulation and analysis.
* matplotlib: For plotting graphs.
* time: For measuring execution time.
# How to Use
* Ensure you have the necessary libraries installed, such as numpy, pandas, matplotlib, and time.
* Prepare your dataset with features (x0 to x8) and the target variable (y).
* Run the provided code for each optimization technique (gradient descent, mini-batch gradient descent, momentum) to observe the convergence behavior and performance.
# Output
The code outputs the convergence behavior of each optimization technique in terms of the cost function over time. It also displays the time taken for each optimization technique to converge.
