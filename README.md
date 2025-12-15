# Comparative Time Series Forecasting for Electricity Consumption

## Overview

This repository hosts a comparative analysis of five distinct methodologies for time series forecasting, applied to historical monthly electricity consumption data (`data/27.csv`). The project's objective is to evaluate model stability and predictive accuracy to produce a robust **7-year (84-month)** out-of-sample forecast.

## Repository Structure

The code and data are organized into the following directories:

* **`code/`**: Contains all Python source code (`.py` files) for data utilities, modeling, and reporting.
* **`results/`**: Reserved for all generated output files (forecast plots and CSVs).
* **`EDA/`**: Contains additional plots for time series analysis.


## Models and Techniques Implemented

The project compares model performance using Un-scaled Root Mean Squared Error (RMSE) on a fixed 24-month test set.

1.  **Classical Statistical Models:**
    * `sarima.py`: Seasonal AutoRegressive Integrated Moving Average (SARIMA) for mean forecasting.
    * `garch.py`: Generalized AutoRegressive Conditional Heteroskedasticity (GARCH) for volatility modeling of SARIMA residuals.
2.  **Statistical Smoothing:**
    * `ets.py`: Exponential Smoothing (Holt-Winters) for recursive forecasting.
3.  **Decomposition:**
    * `FBprophet.py`: Prophet model for robust trend and seasonality capture.
4.  **Deep Learning:**
    * `lstm.py`: Long Short-Term Memory (LSTM) network optimized for capturing non-linear seasonality.

## Setup and Execution

### Prerequisites

Python 3.8+ and the libraries listed in `requirements.txt`.

### Installation

```bash
pip install -r requirements.txt
