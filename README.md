# QUANT PROJECT 2: REGRESSION MODELING FOR STOCK PRICE PREDICTION

This project applies and compares multiple regression models to predict the next-day closing price of Apple Inc. (AAPL) using historical market data. Models include Ordinary Least Squares (OLS), Ridge Regression, Lasso Regression, and Elastic Net Regression.

The full pipeline includes data collection, feature engineering, regression modeling, statistical diagnostics, and forward testing on future data.

---

## OBJECTIVE

Predict the next-day closing price of AAPL stock using lagged values and market indices.

The project begins with OLS regression and extends to regularized models (Ridge, Lasso, ElasticNet) to evaluate improvements in predictive performance and model robustness.

---

## SETUP INSTRUCTIONS

## ```bash
git clone https://github.com/rithik279/Project2_linear_regression_stock_prices
cd Project2_linear_regression_stock_prices
pip install -r requirements.txt

PROJECT PIPELINE

## 1. DATA COLLECTION
Source: Yahoo Finance via yfinance

Tickers: AAPL, AMZN, MSFT, QQQ, ^GSPC

Range: 2020-01-01 to 2024-12-31 (training), 2025-01-01 to 2025-07-29 (testing)

## 2. FEATURE ENGINEERING
Lagged price features: (t-1) values for all tickers

Moving average features: 5-day MA for each ticker

Target variable: AAPL closing price shifted by -1 (next-day)

## 3. MODELING
OLS Regression using statsmodels.OLS

Predictors: AAPL(t-1), ^GSPC(t-1)

Output: Coefficients, p-values, R², full statistical summary

Lasso Regression

Includes L1 regularization to perform feature selection

Ridge Regression

Includes L2 regularization to mitigate multicollinearity

Elastic Net Regression

Combines L1 and L2 penalties to balance sparsity and stability

Each model is trained using identical features and evaluated using R², mean squared error, and visual comparisons of predicted vs. actual prices.

## 4. PERFORMANCE EVALUATION
Prediction accuracy metrics: R², Mean Squared Error

Visual diagnostics:

Actual vs. Predicted Price Plot

Residual vs. Fitted Scatter Plot

Histogram and QQ plot of residuals

Statistical checks:

Linearity: Pair plots

Homoscedasticity

Multicollinearity: Variance Inflation Factor (VIF)

Normality: QQ plot and histogram

Autocorrelation: Durbin-Watson statistic

## 5. TESTING ON FUTURE DATA
Forward testing on January–July 2025

Models applied to new data with consistent feature engineering

Predictions compared visually and numerically

## VISUALIZATIONS
Line plots: Actual vs. Predicted Prices (all models)

Residual diagnostics (scatter, histogram, QQ plot)

Pair plots of features

Coefficient visualizations (Lasso, Ridge, ElasticNet)

## KEY FINDINGS
OLS provides transparency but is sensitive to multicollinearity and overfitting.

Lasso eliminates weak predictors through L1 penalty, offering a sparse model.

Ridge stabilizes coefficient estimates where predictors are correlated.

ElasticNet balances sparsity and regularization strength, often outperforming Lasso and Ridge when tuned properly.

Regularized models demonstrated better generalization on unseen data in 2025 compared to OLS.

## REQUIREMENTS
All libraries can be installed via:

bash
Copy
Edit
pip install -r requirements.txt
Major dependencies:

yfinance

pandas

matplotlib

seaborn

statsmodels

scikit-learn

numpy

## FILES INCLUDED
notebooks/linear_model_dev.ipynb: Jupyter notebook with full analysis and model comparisons

logic/: Python modules for each regression model and data preprocessing

requirements.txt: Python dependencies

README.md: Project overview and documentation

## AUTHOR NOTES
This project is part of a broader quant research effort to build statistical modeling intuition, compare regression techniques, and develop reproducible workflows for financial prediction tasks. It also lays the groundwork for future builds incorporating more advanced models and deployment methods.

## TO-DO / NEXT STEPS
Add rolling standard deviation (volatility) as a new feature

Include additional market indicators (e.g., VIX, trading volume)

Expand to tree-based models (e.g., Random Forest, Gradient Boosting)

Deploy via Streamlit or FastAPI for interactive predictions

© 2025 Rithik Singh — Quant Finance Project Series