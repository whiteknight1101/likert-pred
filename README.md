# Likert Score Prediction for Microsuturing Images

The problem statement is predicting Likert scores (between 1-9) given to Microsuturing performed by trainees by senior doctors. I have implemented linear regression for carrying out this task.

I have carried out the task in three ways:
1. Vanilla Linear Regression from Scratch: I have implemented linear regression from scratch in Python, and used the gradient descent method for optimizing the loss function.
2. Ridge Regression: I have penalized high value weights by adding a term $\lambda ||w||^2$ to the loss function. This helps in regularization and prevents overfitting.
3. Scikit-Learn: I also used the scikit learn library for implementing linear regressions.

I have also visualized the data using Matplotlib.
