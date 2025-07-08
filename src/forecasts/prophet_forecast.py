from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import itertools
import numpy as np
from src.utils.aggregate_timeseries import aggregate_timeseries
from src.utils.assess_backtests import assess_backtest


def merge_target_and_regressors(target_df, date_col, target_col, regressor_dfs=None, regressor_cols=None):
    """
    Merge a main target DataFrame with any number of regressor DataFrames on the date column.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame with [date_col, target_col].
    date_col : str
        Name of the date column.
    target_col : str
        Name of the target column.
    regressor_dfs : list of pd.DataFrame, optional
        List of DataFrames, each with [date_col, regressor_col].
    regressor_cols : list of str, optional
        List of regressor column names (same order as regressor_dfs).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns [date_col, target_col, regressor_1, regressor_2, ...]
    """
    merged = target_df[[date_col, target_col]].copy()
    if regressor_dfs and regressor_cols:
        for df, col in zip(regressor_dfs, regressor_cols):
            merged = merged.merge(df[[date_col, col]], on=date_col, how='outer')
            merged[col] = merged[col].fillna(0)
    merged = merged.sort_values(date_col).reset_index(drop=True)
    return merged

def prophet_grid_search(df, target_col, date_col, param_grid, use_regressors=False):
    """
    Runs a grid search over Prophet hyperparameters for a single time series.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the date_col and target_col columns.
    target_col : str
        Name of the column containing the target variable for forecasting.
    date_col : str
        Name of the column containing the datetime values.
    param_grid : dict
        Dictionary where keys are Prophet parameter names and values are lists of values to try.

    Returns
    -------
    pandas.DataFrame
        DataFrame with each combination of parameters tried and the resulting MAPE.
    """
    regressor_cols = [col for col in df.columns if col not in [date_col, target_col]] if use_regressors else []
    all_cols = [date_col, target_col] + regressor_cols
    df = df[all_cols].rename(columns={date_col: 'ds', target_col: 'y'})
    df = df.sort_values('ds')
    # Only use rows with non-missing y for train/test split
    observed = df[df['y'].notna()]
    last_6 = observed['ds'].sort_values().unique()[-6:]
    train = observed[~observed['ds'].isin(last_6)]
    test = observed[observed['ds'].isin(last_6)]
    results = []
    keys = param_grid.keys()
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(keys, params))
        model = Prophet(
            changepoint_prior_scale=param_dict.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=param_dict.get('seasonality_prior_scale', 10.0),
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        model.add_seasonality(
            name='custom_yearly',
            period=365.25,
            fourier_order=param_dict.get('fourier_order', 10)
        )
        for col in regressor_cols:
            model.add_regressor(col)
        try:
            model.fit(train)
            # Predict only for train + test dates
            future = df[df['ds'].isin(train['ds']) | df['ds'].isin(test['ds'])][['ds'] + regressor_cols].sort_values('ds')
            forecast = model.predict(future)
            preds = forecast[['ds', 'yhat']].set_index('ds').loc[test['ds']].reset_index()
            mape = mean_absolute_percentage_error(test['y'], preds['yhat'])
        except Exception as e:
            mape = float('inf')
        param_dict['mape'] = mape
        results.append(param_dict)
    return pd.DataFrame(results)


def prophet_forecast(
    df,
    target_col,
    date_col='Month',
    horizon=4,
    param_grid=None,
    use_regressors=False,
    verbose=False,
    backtest=False
):
    """
    Prophet forecasting (single series). Returns a dict with forecast, best params, gridsearch MAPE,
    and, if backtest=True, backtest metrics.
    """

    working_df = df.copy()

    # If backtest, mask last h values as NaN and save them for metrics
    if backtest:
        non_nan = working_df[working_df[target_col].notna()]
        mask_idx = non_nan.index[-horizon:]
        backtest_true = working_df.loc[mask_idx, [date_col, target_col]].copy()
        working_df.loc[mask_idx, target_col] = np.nan
    else:
        backtest_true = None

    if param_grid is None:
        param_grid = {
            'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.3, 0.5, 1],
            'seasonality_prior_scale': [5, 10, 15, 20],
            'fourier_order': [5, 10, 15, 20]
        }

    regressor_cols = [col for col in working_df.columns if col not in [date_col, target_col]] if use_regressors else []
    all_cols = [date_col, target_col] + regressor_cols

    if horizon is None:
        horizon = working_df[target_col].isna().sum()

    # --- GRID SEARCH ---
    grid_results = prophet_grid_search(working_df, target_col, date_col, param_grid, use_regressors)
    best_params = grid_results.sort_values('mape').iloc[0]
    best_params_dict = {k: (float(v) if isinstance(v, (np.generic, np.floating, np.integer)) else v)
                        for k, v in best_params.items() if k in param_grid}

    prophet_df = working_df[all_cols].rename(columns={date_col: 'ds', target_col: 'y'}).sort_values('ds')
    model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    model.add_seasonality(
        name='custom_yearly',
        period=365.25,
        fourier_order=int(best_params['fourier_order'])
    )
    for col in regressor_cols:
        model.add_regressor(col)
    model.fit(prophet_df[prophet_df['y'].notna()])

    last_train_date = prophet_df.loc[prophet_df['y'].notna(), 'ds'].max()
    future_months = pd.date_range(last_train_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
    if use_regressors and regressor_cols:
        future_regressors = working_df[working_df[date_col].isin(future_months)][[date_col] + regressor_cols].copy()
        future_regressors = future_regressors.rename(columns={date_col: 'ds'}).sort_values('ds')
        forecast = model.predict(future_regressors)
    else:
        future_dates = pd.DataFrame({'ds': future_months})
        forecast = model.predict(future_dates)

    forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': date_col, 'yhat': 'forecast'}).reset_index(drop=True)

    # --- Prepare output dictionary ---
    output = {
        "forecast_df": forecast_df,
        "best_params": dict(best_params_dict),
        "best_gridsearch_mape": float(best_params['mape']),
    }

    # Optionally print betas and Prophet components
    if verbose and use_regressors and regressor_cols:
        beta_values = model.params['beta']
        print("\nRegressor coefficients (betas):")
        for name, val in zip(regressor_cols, beta_values[0]):
            print(f"  {name}: {val:.4f}")
        try:
            from prophet.plot import plot_components
            print("\nProphet component plots (including regressor effect):")
            plot_components(model, forecast)
        except ImportError:
            print("(Could not import Prophet plot_components)")

    # Backtest assessment (metrics)
    if backtest:
        forecast_df = forecast_df.merge(backtest_true, on=date_col, how='left', suffixes=('', '_actual'))
        output["forecast_df"] = forecast_df
        metrics = assess_backtest(forecast_df, target_col)
        output["metrics"] = metrics

    return output




# df = pd.read_csv('../../Data/processed/final_merge.csv')
#
# agg_df = aggregate_timeseries(df,
#                               value_col='Weight Consumed',
#                               split=False,
#                               split_col='National')
#
# agg_df['Month'] = pd.to_datetime(agg_df['Month'])
#
# oil_reg = pd.read_csv("../../Data/processed/oil_monthly_lag3.csv")
# oil_reg['Month'] = pd.to_datetime(oil_reg['Month'])
#
# test_params = {
#             'changepoint_prior_scale': [0.3, 0.5],
#             'seasonality_prior_scale': [10, 15],
#             'fourier_order': [5, 10]
#         }
#
# # merged_df = merge_target_and_regressors(
# #     target_df=agg_df,
# #     date_col='Month',
# #     target_col='Weight Consumed',
# #     regressor_dfs=[oil_reg],
# #     regressor_cols=['Oil Price Lag3']
# # )
#
# forecast = prophet_forecast(
#     agg_df,
#     target_col='Weight Consumed',
#     date_col='Month',
#     param_grid=test_params,
#     horizon=4,
#     use_regressors=False,
#     verbose=False,
#     backtest=True
# )
#
# print(forecast)
