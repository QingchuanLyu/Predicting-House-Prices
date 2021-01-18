# Predicting-Iowa-House-Prices
This project predicted house prices in Ames, Iowa with 79 features (2006-2010). The training set had 1460 observations and the test set had 1459 observations.
## Updates (01/17/2021)
* Engineered features for tree algorithms and linear regression separately. Used label-encoders for trees and one-hot encoders for linear regression with regularization
* Checked whether missing values are missing (completely) at random
* PCA helped avoid overfitting by combining collinear features for linear models; Random Forest with PCA performed slightly better (0.17 vs 0.18 RMSE) but needed many more trees in each iteration (99 vs 50 trees).

These updates decreased RMSE from 0.24 to 0.18 with Random Forest, and from 0.33 to 0.13 with Elastic Net embedded with PCA.
