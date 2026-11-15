# House Prices: Advanced Regression Techniques

Predicting house sale prices on the Kaggle [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) dataset: 1,460 houses sold in Ames, Iowa, described by 79 features.

## What I've done so far

- **EDA:** distributions, missing values, correlations and skew.
- **Baseline Linear Regression** (numeric features only): ~38.7k RMSE.
- **Ridge with a preprocessing pipeline:** ~34k RMSE.
- **Random Forest:** ~30k RMSE.
- **Gradient Boosting:** ~27k RMSE (0.1315 on the log scale with a log target).

## Latest update

I came back to the project after a long break. After looking at the Ridge residual plot, I decided to try Ridge again with the outliers removed and a log-transformed target.

The residual plot showed:
- The error grows as the price goes up.
- There is a curve in the residuals.
- One house got a predicted price of about $0.
- The features are not additive: their effects are percentages of the price.
- The outliers are obvious, and they hurt linear models.

So I removed the outliers (`GrLivArea > 4000` and `SalePrice < 300000`) and applied a log transform to the target.

## Next steps

- Compare the new Ridge results against the old ones.
- Scale features and transform skewed ones.
- Try Lasso and blend models.

## Notebook

`house_prices.ipynb`
