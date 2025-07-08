import time

start_time = time.time()

from datetime import datetime

import pandas as pd
from src.utils.forecast_pipeline import run_forecast_pipeline

df = pd.read_csv('../Data/processed/final_merge.csv')
df['Month'] = pd.to_datetime(df['Month'])

oil = pd.read_csv('../Data/processed/oil_monthly_lag3.csv')
oil['Month'] = pd.to_datetime(oil['Month'])

demand = df.groupby('Month', as_index=False)['Weight Sold'].sum()

param_grid = {
            'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.3, 0.5, 1],
            'seasonality_prior_scale': [5, 10, 15, 20],
            'fourier_order': [5, 10, 15, 20]
        }

regressor_sources = {
    "Oil Price Lag3": oil,
    "Weight Sold": demand
}

reg_dict = {
    ("Oil Price Lag3", "Weight Sold"): ["Oil Price Lag3", "Weight Sold"],
    ("Oil Price Lag3",): ["Oil Price Lag3"],
    ("Weight Sold",): ["Weight Sold"]
}


results = []

for model in ['prophet', 'ma']:
    for target in ['Weight Sold', 'Weight Consumed']:
        for com in [None, 3, 6, 9, 12]:
            for h in [2, 4, 6]:
                windows = [4, 6, 8, 10] if model == 'ma' else [None]

                if model == 'ma':
                    regressor_configs = [(None, None)]
                    use_regressors = False
                elif target == 'Weight Sold':
                    regressor_configs = [([oil], ['Oil Price Lag3'])]
                    use_regressors = True
                else:
                    regressor_configs = [
                        ([regressor_sources[name] for name in reg_col_names], list(reg_col_names))
                        for reg_col_names in reg_dict.keys()
                    ]
                    use_regressors = True

                for regressor_dfs, regressor_cols in regressor_configs:
                    for w in windows:
                        fct = run_forecast_pipeline(
                            df=df,
                            target_col=target,
                            model=model,
                            date_col='Month',
                            split=False,
                            split_col='National',
                            use_regressors=use_regressors,
                            regressor_dfs=regressor_dfs,
                            regressor_cols=regressor_cols,
                            window=w,
                            param_grid=param_grid,
                            horizon=h,
                            backtest=True,
                            plot=False,
                            save=False,
                            output_dir=None,
                            verbose=False,
                            cutoff_months=com
                        )
                        results.append(fct)


results_df = pd.DataFrame(results)
metrics_df = pd.json_normalize(results_df['metrics'])
summary_df = pd.concat([results_df.drop('metrics', axis=1), metrics_df], axis=1)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_df.to_csv(f'../Data/processed/Forecasts/forecast_summary_{ts}.csv')

end_time = time.time()
print("Script took {:.2f} seconds to run.".format(end_time - start_time))
