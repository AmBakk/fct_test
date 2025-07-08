import matplotlib.pyplot as plt
import pandas as pd
from src.forecasts.prophet_forecast import merge_target_and_regressors, prophet_forecast
from src.utils.aggregate_timeseries import aggregate_timeseries
from src.forecasts.MA_Forecast import ma_forecast



def plot_forecast(
        df,
        date_col,
        target_col,
        forecast_df,
        forecast_date_col=None,
        forecast_col='forecast',
        model_name='Forecast',
):
    """
    Plots full historical data with forecast appended as a dotted line.
    In all cases, draws a join (dotted, blue) from the last training actual to the first forecasted value.
    """
    plt.figure(figsize=(12, 6))
    # Plot historical actuals
    plt.plot(df[date_col], df[target_col], label="Actual", color="black", linewidth=2)

    # Join last training actual to first forecast value (always)
    train_actuals = df[df[target_col].notna() & (df[date_col] < forecast_df[forecast_date_col or date_col].min())]
    if not train_actuals.empty:
        last_train_actual_date = train_actuals[date_col].max()
        last_train_actual_value = train_actuals.loc[train_actuals[date_col] == last_train_actual_date, target_col].values[0]
        first_forecast_date = forecast_df[forecast_date_col or date_col].min()
        first_forecast_value = forecast_df.loc[forecast_df[forecast_date_col or date_col] == first_forecast_date, forecast_col].values[0]
        plt.plot(
            [last_train_actual_date, first_forecast_date],
            [last_train_actual_value, first_forecast_value],
            linestyle='dotted', color="tab:blue"
        )

    # Plot the forecasted values
    plt.plot(
        forecast_df[forecast_date_col or date_col],
        forecast_df[forecast_col],
        linestyle='dotted', color="tab:blue", label=model_name, marker='o'
    )

    # Y axis always starts at 0
    plt.ylim(bottom=0)

    # X-axis limits: use min/max from both actual and forecast, pad a little
    min_date = df[df[target_col].notna()][date_col].min()
    max_date = forecast_df[forecast_date_col or date_col].max()
    plt.xlim(min_date, max_date + pd.DateOffset(months=1))

    # Clean x-axis
    plt.ylabel(target_col)
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    # Reduce label clutter
    all_xticks = pd.date_range(min_date, max_date, freq='MS')
    N = max(len(all_xticks) // 15, 1)
    plt.gca().set_xticks(all_xticks[::N])
    plt.tight_layout()
    plt.show()

#
# #
# df = pd.read_csv('../../Data/processed/final_merge.csv')
#
# agg_df = aggregate_timeseries(df,
#                               value_col='Weight Consumed',
#                               split=True,
#                               split_col='National')
#
# agg_df['Month'] = pd.to_datetime(agg_df['Month'])
#
# oil_reg = pd.read_csv("../../Data/processed/oil_monthly_lag3.csv")
# oil_reg['Month'] = pd.to_datetime(oil_reg['Month'])
#
# test_params = {
#             'changepoint_prior_scale': [0.5],
#             'seasonality_prior_scale': [10],
#             'fourier_order': [5]
#         }
#
# merged_df = merge_target_and_regressors(
#    target_df=agg_df,
#    date_col='Month',
#    target_col='Weight Consumed',
#    regressor_dfs=[oil_reg],
#    regressor_cols=['Oil Price Lag3']
# )
#
# forecast = prophet_forecast(
#    merged_df,
#    target_col='Weight Consumed',
#    date_col='Month',
#    param_grid=test_params,
#    horizon=6,
#    use_regressors=False,
#    verbose=True,
#    split=False,
#    split_col=None,
#    backtest=True
# )
#
# # forecast = ma_forecast(agg_df, value_col='Weight Consumed', date_col='Month', window=4, forecast_periods=4)
#
# plot_forecast(
#     df=agg_df,
#     date_col='Month',
#     target_col='Weight Consumed',
#     forecast_df=forecast,
#     model_name='Prophet Forecast'
# )