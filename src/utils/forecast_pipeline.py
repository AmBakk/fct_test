import traceback
from datetime import datetime

import pandas as pd
from src.utils.aggregate_timeseries import aggregate_timeseries
from src.forecasts.prophet_forecast import merge_target_and_regressors, prophet_forecast
from src.forecasts.MA_Forecast import ma_forecast
from src.utils.plot_forecasts import plot_forecast
from src.utils.assess_backtests import assess_backtest


def run_forecast_pipeline(
    df, target_col, model,
    date_col='Month',
    split=False, split_col=None,
    use_regressors=False, regressor_dfs=None, regressor_cols=None,
    window=4, param_grid=None, horizon=4,
    backtest=False, plot=True, save=False,
    output_dir=None, verbose=True,
    cutoff_months=None
):
    run_id = f"{model}_{target_col}_{horizon}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    metrics = None
    best_params = None
    forecast_sample = None
    forecast_path = None
    plot_path = None
    error = None

    last_date = df[df[target_col].notna()][date_col].max()

    if cutoff_months is not None:
        cutoff_date = (pd.to_datetime(last_date) - pd.DateOffset(months=cutoff_months))
        working_df = df[df[date_col] <= cutoff_date].copy()
    else:
        working_df = df.copy()

    try:
        # 1. Aggregate
        agg_df = aggregate_timeseries(working_df, target_col, date_col, split, split_col)

        # 2. Merge regressors if needed
        if model == 'prophet' and use_regressors and regressor_dfs is not None:
            merged_df = merge_target_and_regressors(
                agg_df, date_col, target_col, regressor_dfs, regressor_cols
            )
        else:
            merged_df = agg_df

        # 3. Run forecast
        if model == 'prophet':
            forecast_result = prophet_forecast(
                merged_df, target_col, date_col, horizon=horizon,
                param_grid=param_grid,
                use_regressors=use_regressors, backtest=backtest, verbose=verbose
            )
            forecast_df = forecast_result['forecast_df'] if isinstance(forecast_result, dict) else forecast_result
            best_params = forecast_result.get('best_params') if isinstance(forecast_result, dict) else None
            metrics = forecast_result.get('metrics') if isinstance(forecast_result, dict) else None
        elif model == 'ma':
            forecast_result = ma_forecast(
                merged_df, target_col, date_col, window=window,
                forecast_periods=horizon,
                backtest=backtest
            )
            forecast_df = forecast_result['forecast_df'] if isinstance(forecast_result, dict) else forecast_result
            metrics = forecast_result.get('metrics') if isinstance(forecast_result, dict) else None
            best_params = {'window': window}
        else:
            raise ValueError("Model not recognized.")

        # 4. Plot
        if plot:
            plot_forecast(
                df=merged_df,
                date_col=date_col,
                target_col=target_col,
                forecast_df=forecast_df,
                model_name=f'{model.title()} Forecast'
            )
        # 5. Save output if needed
        if save and output_dir is not None:
            forecast_path = f"{output_dir}/{run_id}_forecast.csv"
            forecast_df.to_csv(forecast_path, index=False)
        # 6. Print summary (optional)
        if verbose:
            print("-" * 30)
            print(f"Model: {model}")
            print(f"Parameters: {param_grid if model == 'prophet' else window}")
            print(f"Horizon: {horizon} | Backtest: {backtest}")
            print(f"Actuals from {merged_df[date_col].min()} to {merged_df[date_col].max()}")
            print(f"Forecast from {forecast_df[date_col].min()} to {forecast_df[date_col].max()}")
            print("-" * 30)

        forecast_json = forecast_df.to_dict(orient='records')
        return {
            'run_id': run_id,
            'model': model,
            'target_col': target_col,
            'window': window if model == 'ma' else None,
            'param_grid': param_grid if model == 'prophet' else None,
            'horizon': horizon,
            'use_regressors': use_regressors,
            'regressor_cols': regressor_cols,
            'backtest': backtest,
            'start_date': str(merged_df[merged_df[target_col].notna()][date_col].min()),
            'end_date': str(merged_df[merged_df[target_col].notna()][date_col].max()),
            'metrics': metrics,
            'best_params': best_params if model == 'prophet' else None,
            'forecast_json': forecast_json,
            'error': error
        }
    except Exception as e:
        print(f"Error in pipeline: {e}")
        traceback.print_exc()
        error = str(e)
        return {
            'run_id': run_id,
            'model': model,
            'target_col': target_col,
            'window': window if model == 'ma' else None,
            'param_grid': param_grid if model == 'prophet' else None,
            'horizon': horizon,
            'use_regressors': use_regressors,
            'regressor_cols': regressor_cols,
            'backtest': backtest,
            'start_date': None,
            'end_date': None,
            'metrics': None,
            'best_params': None,
            'forecast_json': None,
            'error': error
        }
