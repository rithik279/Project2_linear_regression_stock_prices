# QUANT PROJECT 2: LINEAR REGRESSION MODELING FOR STOCK PRICE PREDICTION

This project builds a linear regression model to predict the next-day closing price of Apple Inc. (AAPL) using historical market data. It includes feature engineering with technical indicators, model training, diagnostics for key regression assumptions, and a forecasting exercise on 2025 data.

---

## OBJECTIVE

Predict the next-day closing price of AAPL stock using lagged values and market indices.

The model is constructed using Ordinary Least Squares (OLS) and evaluated against five core assumptions of linear regression to ensure statistical validity.

---

## PROJECT PIPELINE

### 1. DATA COLLECTION
- Source: Yahoo Finance
- Tickers: AAPL, AMZN, MSFT, QQQ, ^GSPC
- Range: 2020-01-01 to 2024-12-31 (training), 2025-01-01 to 2025-07-29 (testing)

### 2. FEATURE ENGINEERING
- Lagged price features: (t-1) values for all tickers
- Moving average features (MA-5)
- Target: AAPL price shifted by -1 (i.e., next-day price)

### 3. MODELING
- Model: statsmodels.OLS()
- Predictors: AAPL(t-1), ^GSPC(t-1)
- Output: Regression coefficients, p-values, R², and full statistical summary

### 4. PERFORMANCE EVALUATION
- Compare predicted vs. actual prices visually using line plots
- Check key assumptions using diagnostics:
  - Linearity: Pair plots
  - Homoscedasticity: Residual vs. Fitted scatter
  - Multicollinearity: VIF values
  - Normality of residuals: Histogram and QQ plot
  - Autocorrelation: Durbin-Watson test

### 5. TESTING ON FUTURE DATA
- New data range: January–July 2025
- Same features engineered
- Prediction performed using the trained model

---

## VISUALIZATIONS

- Actual vs. Predicted Line Charts
- Residual vs. Fitted Scatterplots
- Histogram and QQ Plot of Residuals
- Seaborn Pairplots

---

## KEY LEARNINGS

- Linear regression offers transparency and interpretability, but struggles with non-linear patterns in stock data.
- While assumptions were reasonably satisfied, the predictive accuracy remains limited.
- For production-grade systems, machine learning models (e.g., XGBoost, LSTM, Random Forests) often outperform traditional regressions in capturing the complex dynamics of financial markets.

---

## REQUIREMENTS


pip install yfinance statsmodels seaborn matplotlib pandas

## FILES INCLUDED
notebooks/linear_model_dev.ipynb – Full code and analysis

README.md – Project overview and methodology

## AUTHOR NOTES
This project is part of a broader quant research portfolio to build intuition for model diagnostics, technical indicators, and statistical rigor in finance. It is also designed to lay the foundation for more advanced ML-based strategies in future builds.

## TO-DO / NEXT STEPS
Add rolling standard deviation (volatility) as a feature

Introduce additional market indicators (VIX, Volume)

Build and compare tree-based regressors (Random Forest, Gradient Boost)

Deploy model via Streamlit for interactive predictions

© 2025 Rithik Singh — Quant Finance Project Series

