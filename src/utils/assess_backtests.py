import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def assess_backtest(
    backtest_df,
    target_col,
    forecast_col='forecast',
    verbose=True
):
    """
    Compute common backtest metrics for time series forecast.
    Returns a dict with MAE, MAPE, RMSE, R2, Bias (ME).
    """
    df = backtest_df[[forecast_col, target_col]].dropna()
    y_true = df[target_col].values
    y_pred = df[forecast_col].values

    results = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Bias': np.mean(y_pred - y_true)
    }

    if verbose:
        print("Backtest metrics:")
        for k, v in results.items():
            if k == 'MAPE':
                print(f"{k}: {v:.2%}")
            else:
                print(f"{k}: {v:.3f}")
    return results