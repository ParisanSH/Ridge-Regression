# Ridge Regression

# Problem Definition:
The ridge regression objective function is:

πin(w) (π β Xw)π(π β Xw) + π wπw.

Where the hyperparameter π controls the amount of regularization. Closed form solution of this function is:

w = (XπX + π I)β1 Xππ.

In this project we are going to find hyperparameter π so that it minimizes the following objective function:

πin(π) (π β πw)π(π β πw).

Where Z is validation data, and T is its target value.

# Dataset:
The UCI Machine Learning Repository is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms. The archive was created as an FTP archive in 1987 by David Aha and fellow graduate students at UC Irvine. Since that time, it has been widely used by students, educators, and researchers all over the world as a primary source of machine learning data sets.

One of the regression datasets of the UCI repository, the real estate dataset, is chosen and split into 70% training, 10% validation, and 20% test data.

# Steps:

β’	Finding optimal π.

β’	Calculating MSE of Ridge regression for training, validation and test data using founded π and L2-norm of its w.

β’	Calculating MSE of Linear regression for training, validation and test data and l2-norm of its w.
