# Ridge Regression

# Problem Definition:
The ridge regression objective function is:

𝑚in(w) (𝑌 − Xw)𝑇(𝑌 − Xw) + 𝜆 w𝑇w.

Where the hyperparameter 𝜆 controls the amount of regularization. Closed form solution of this function is:

w = (X𝑇X + 𝜆 I)−1 X𝑇𝑌.

In this project we are going to find hyperparameter 𝜆 so that it minimizes the following objective function:

𝑚in(𝜆) (𝑇 − 𝑍w)𝑇(𝑇 − 𝑍w).

Where Z is validation data, and T is its target value.

# Dataset:
The UCI Machine Learning Repository is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms. The archive was created as an FTP archive in 1987 by David Aha and fellow graduate students at UC Irvine. Since that time, it has been widely used by students, educators, and researchers all over the world as a primary source of machine learning data sets.

One of the regression datasets of the UCI repository, the real estate dataset, is chosen and split into 70% training, 10% validation, and 20% test data.

# Steps:
•	Finding optimal 𝜆.
•	Calculating MSE of Ridge regression for training, validation and test data using founded 𝜆 and L2-norm of its w.
•	Calculating MSE of Linear regression for training, validation and test data and l2-norm of its w.
