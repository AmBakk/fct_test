import pandas as pd
import numpy as np
from src.utils.aggregate_timeseries import aggregate_timeseries
from src.utils.assess_backtests import assess_backtest


def ma_forecast(
        df,
        target_col,
        date_col='Month',
        window=4,
        forecast_periods=4,
        backtest=False
):
    """
    Simple non-recursive moving average forecast with optional backtest mode.
    Returns dict: forecast DataFrame, params, and (if backtest) metrics.
    """
    agg_df = df.sort_values(date_col).copy()
    freq = 'MS'
    params = {'window': window, 'forecast_periods': forecast_periods}

    # -------- BACKTEST MODE --------
    if backtest:
        train = agg_df.iloc[:-forecast_periods]
        test = agg_df.iloc[-forecast_periods:]
        test_dates = test[date_col].values

        series = train[target_col].tolist()
        forecasts = []
        for _ in range(forecast_periods):
            window_vals = [x for x in series[-window:] if pd.notna(x)]
            next_forecast = np.mean(window_vals) if window_vals else np.nan
            forecasts.append(next_forecast)

        forecast_df = pd.DataFrame({
            date_col: test_dates,
            'forecast': forecasts,
            target_col: test[target_col].values
        })

        # Metrics
        metrics = assess_backtest(forecast_df, target_col=target_col, forecast_col='forecast', verbose=False)

        return {
            "model": "ma",
            "parameters": params,
            "metrics": metrics,
            "forecast_df": forecast_df
        }

    # -------- FORECAST FUTURE --------
    else:
        last_date = agg_df[date_col].max()
        series = agg_df[target_col].tolist()
        forecasts = []
        for _ in range(forecast_periods):
            window_vals = [x for x in series[-window:] if pd.notna(x)]
            next_forecast = np.mean(window_vals) if window_vals else np.nan
            forecasts.append(next_forecast)
            # Do not append forecast—non-recursive MA
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1),
                                     periods=forecast_periods, freq=freq)
        forecast_df = pd.DataFrame({
            date_col: future_dates,
            'forecast': forecasts
        })

        return {
            "model": "ma",
            "parameters": params,
            "forecast_df": forecast_df
        }

# df = pd.read_csv('../../Data/processed/final_merge.csv')
#
# agg_df = aggregate_timeseries(df, 'Weight Consumed')
#
# ma_fct = ma_forecast(agg_df, 'Weight Consumed', backtest=True)
#
# print(ma_fct)
