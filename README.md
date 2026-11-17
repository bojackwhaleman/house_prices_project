# House Prices: Advanced Regression Techniques

Predicting house sale prices on the Kaggle [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) dataset: 1,460 houses sold in Ames, Iowa, described by 79 features. The goal of this project is to learn, so the notebook focuses on understanding why models behave the way they do.

## What I've done so far

- **EDA:** distributions, missing values, correlations and skew.
- **First models** (before the fixes below), cross-validated RMSE:
  - Linear Regression baseline (numeric features only): ~38.7k
  - Ridge with a preprocessing pipeline: ~34k
  - Random Forest: ~30k
  - Gradient Boosting: ~27k

## Improving the models

I came back to the project after a long break and went through the Ridge residual plot. It showed that the error grows with the price, that the residuals curve, and that one house got a predicted price of about $0. From that I made these changes:

1. **Removed the outliers:** very large houses that sold cheap (`GrLivArea > 4000` and `SalePrice < 300000`) and a few houses with huge garages that sold cheap.
2. **Log-transformed the target** with `TransformedTargetRegressor`, because errors and effects scale with the price.
3. **Fixed a pipeline bug:** the `ColumnTransformer` silently dropped every column not listed in it, so 27 categorical columns (including `Neighborhood`, `SaleCondition` and `KitchenQual`) were never used. The model went from 120 to 301 input columns, built from all 79 original features. A check now confirms that no column is left out.
4. **Reviewed how missing values are filled:** for text columns, a missing value means "the house doesn't have it" (no garage, no alley) and is filled with `None`. `Electrical` is a real gap and is filled with the most common value. `MasVnrArea` and `GarageYrBlt` are filled with 0.

## Results (5-fold cross-validation, outliers removed, log target)

| Model | RMSE on log scale | RMSE ($) |
|---|---|---|
| Ridge | 0.1249 | ~22.9k |
| Gradient Boosting | **0.1153** | ~21.9k |

The log-scale RMSE is the metric Kaggle uses for this competition (lower is better).

## What I learned

- A residual plot shows what a model is missing: a funnel means the errors scale with the price, and a curve means the straight-line fit isn't enough.
- Outliers that contradict the data (huge houses that sold cheap) hurt linear models, but houses that are only unusual or expensive should stay.
- Always check that a pipeline covers every column. `ColumnTransformer` drops anything not listed.
- Compare models on the same data and folds, or the scores aren't comparable.

## Next steps

- Find the best Ridge alpha with a proper search instead of trying values by hand.
- Scale the numeric features and transform the skewed ones.
- Encode the ordered quality columns (Po < Fa < TA < Gd < Ex) as numbers.
- Add a few meaningful combined features, such as total square footage and house age.
- Try Lasso and blend models.

## Notebook

`house_prices.ipynb`
