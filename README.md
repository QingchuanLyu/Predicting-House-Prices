# Predicting-Iowa-House-Prices
This project predicted house prices in Ames, Iowa with 79 features (2006-2010). The training set had 1460 observations and the test set had 1459 observations.
## Updates (12/20/2020)
* Engineered features for tree algorithms and linear regression separately. Used label-encoders for trees and one-hot encoders for linear regression with regularization
* Checked whether missing values are missing (completely) at random
* Figured out PCA reduced prediction accuracy for all models, but helped avoid overfitting by combining collinear features for linear models.

These updates decreased RMSE from 0.18 to 0.06 with Random Forest, and from 0.12 to 0.11 with Elastic Net.
