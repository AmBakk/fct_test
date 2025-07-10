import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import ast  # Use ast for safe literal evaluation
import pathlib
import re  # Import regex library for timestamp parsing

# --- Configuration & Styling ---
st.set_page_config(layout="wide", page_title="Forecast Comparison Dashboard")

st.title("🔎 Forecast Comparison Dashboard")
st.markdown("""
Use the sidebar to filter forecasts. Then, in the main area, select one or more runs to compare their metrics and plots.
All plots show the full history of actuals, with the forecast period highlighted for direct comparison.
""")

# --- Path Configuration ---
# Get the directory of the current script
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Construct absolute paths
# --- IMPORTANT: UPDATE FILENAMES HERE ---
LOGO_PATH = None
FORECAST_PATH = PROJECT_ROOT / "Data/processed/Forecasts/forecast_summary_20250709_180322.csv"
ACTUALS_PATH = PROJECT_ROOT / "Data/processed/final_merge.csv"



# ------------------------------------

# --- Data Loading and Caching ---
@st.cache_data
def load_data(forecast_path, actuals_path):
    # --- Load and Standardize Forecasts DF ---
    try:
        forecasts_df = pd.read_csv(forecast_path)
    except FileNotFoundError:
        st.error(f"Forecast file not found. Checked path: {forecast_path}")
        return None, None

    forecasts_df['target_col'] = forecasts_df['target_col'].replace({
        'Sales': 'Sales (lbs)', 'Consumption': 'Consumption (lbs)'
    })

    metric_cols = ['MAE', 'MAPE', 'RMSE', 'R2', 'Bias']
    for col in metric_cols:
        if col in forecasts_df.columns:
            forecasts_df[col] = pd.to_numeric(
                forecasts_df[col].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            )
            if col == 'MAPE' and not forecasts_df[col].empty and forecasts_df[col].max() > 1:
                forecasts_df[col] /= 100.0

    forecasts_df['window'] = forecasts_df['window'].fillna('N/A').astype(str)

    def create_regressor_str(row):
        if not row['use_regressors']: return 'No Regressors'
        try:
            cols = ast.literal_eval(row['regressor_cols'])
            return ', '.join(cols) if cols else 'No Regressors'
        except:
            return str(row['regressor_cols'])

    forecasts_df['regressor_str'] = forecasts_df.apply(create_regressor_str, axis=1)

    def parse_forecast_json(json_like_str):
        if pd.isna(json_like_str): return pd.DataFrame()
        try:
            processed_str = re.sub(r"Timestamp\('([^']*)'\)", r"'\1'", json_like_str)
            data = ast.literal_eval(processed_str)
            df = pd.DataFrame(data)
            df['Month'] = pd.to_datetime(df['Month'])
            df.rename(columns={'Sales': 'Sales (lbs)', 'Consumption': 'Consumption (lbs)'}, inplace=True)
            return df
        except:
            return pd.DataFrame()

    forecasts_df['forecast_df'] = forecasts_df['forecast_json'].apply(parse_forecast_json)

    forecasts_df['forecast_start_date'] = forecasts_df['forecast_df'].apply(
        lambda df: df['Month'].min() if not df.empty else pd.NaT
    )
    forecasts_df.dropna(subset=['forecast_start_date'], inplace=True)
    forecasts_df['forecast_start_date_str'] = forecasts_df['forecast_start_date'].dt.strftime('%Y-%m-%d')
    forecasts_df.sort_values('forecast_start_date', ascending=False, inplace=True)

    # --- NEW: Create a truly unique key for each row ---
    # We combine run_id with other differentiating columns.
    # The first (unnamed) column is a unique index from the original CSV write.
    # We will use it to guarantee uniqueness.
    if 'Unnamed: 0' in forecasts_df.columns:
        forecasts_df['unique_key'] = forecasts_df['run_id'].astype(str) + '_' + forecasts_df['Unnamed: 0'].astype(str)
    else:
        # Fallback in case the unnamed column is not present
        forecasts_df['unique_key'] = forecasts_df.index.astype(str) + '_' + forecasts_df['run_id'].astype(str)

    # --- FIX: Remove logical duplicates (Now redundant but safe to keep) ---
    cols_to_check_duplicates = [
        'model', 'target_col', 'window', 'horizon',
        'regressor_str', 'forecast_start_date_str'
    ]
    forecasts_df.drop_duplicates(subset=cols_to_check_duplicates, keep='first', inplace=True)

    # --- Load and Standardize Actuals DF ---
    try:
        actuals_df = pd.read_csv(actuals_path)
    except FileNotFoundError:
        st.error(f"Actuals file not found at: {actuals_path}")
        return None, None

    actuals_df['Month'] = pd.to_datetime(actuals_df['YearMonth'])
    rename_map = {'Sales': 'Sales (lbs)', 'Consumption': 'Consumption (lbs)'}
    cols_to_rename = {old: new for old, new in rename_map.items() if old in actuals_df.columns}
    if cols_to_rename: actuals_df.rename(columns=cols_to_rename, inplace=True)
    actuals_df.set_index('Month', inplace=True)

    return forecasts_df, actuals_df


# --- Plotting Function (No changes needed from last working version) ---
def plot_forecast(actuals_df, forecast_run, target_col):
    # This function is the same as the last version, with the addition of date filtering.
    fig = go.Figure()
    forecast_df = forecast_run['forecast_df'].copy()
    if forecast_df.empty: return fig

    forecast_start_date = forecast_df['Month'].min()
    forecast_end_date = forecast_df['Month'].max()
    last_actual_date = forecast_start_date - pd.DateOffset(months=1)

    # --- START OF NEW CODE ---
    # Apply target-specific date limits to the actuals data for plotting
    plot_actuals_df = actuals_df.copy()

    if target_col == 'Sales (lbs)':
        start_date = '2022-02-01'
        end_date = '2025-05-31'
        plot_actuals_df = plot_actuals_df.loc[start_date:end_date]
    elif target_col == 'Consumption (lbs)':
        start_date = '2019-01-01'
        end_date = '2025-02-28'
        plot_actuals_df = plot_actuals_df.loc[start_date:end_date]

    # Truncate to the forecast end date (this happens after the above filter)
    plot_actuals_df = plot_actuals_df[plot_actuals_df.index <= forecast_end_date]

    if last_actual_date not in plot_actuals_df.index:  # Check against the filtered actuals
        continuous_forecast_df = forecast_df
        continuous_actuals_df = forecast_df
    else:
        last_actual_row = plot_actuals_df.loc[[last_actual_date]].reset_index()  # Use filtered actuals
        last_actual_value = last_actual_row.iloc[0][target_col]
        connection_point = pd.DataFrame(
            [{'Month': last_actual_date, 'forecast': last_actual_value, target_col: last_actual_value}])
        continuous_forecast_df = pd.concat(
            [connection_point[['Month', 'forecast']], forecast_df[['Month', 'forecast']]], ignore_index=True)
        continuous_actuals_df = pd.concat([connection_point[['Month', target_col]], forecast_df[['Month', target_col]]],
                                          ignore_index=True)

    fig.add_trace(
        go.Scatter(x=plot_actuals_df.index, y=plot_actuals_df[target_col], mode='lines', name='Actuals (Full History)',
                   line=dict(color='#CCCCCC', width=2)))
    fig.add_trace(go.Scatter(x=continuous_actuals_df['Month'], y=continuous_actuals_df[target_col], mode='lines',
                             name='Actuals (During Forecast)', line=dict(color='#FFA500', width=2.5)))
    fig.add_trace(go.Scatter(x=continuous_forecast_df['Month'], y=continuous_forecast_df['forecast'], mode='lines',
                             name=f"Forecast", line=dict(color='#00E5B4', dash='dash', width=2.5)))

    fig.add_vline(x=last_actual_date, line_width=2, line_dash="dot", line_color="white", opacity=0.8)
    fig.add_annotation(x=last_actual_date, y=1.05, yref="paper", text="Forecast Start", showarrow=False,
                       xanchor="center", font=dict(color="white", size=12))

    fig.update_layout(
        title=f"Forecast vs. Actuals",
        xaxis_title="Month", yaxis_title=target_col, legend_title="Legend",
        template="plotly_dark", font=dict(color="white"),
        legend=dict(bgcolor='rgba(0,0,0,0.3)', y=0.99, x=0.99, yanchor='top', xanchor='right')
    )
    return fig


# --- Main App Logic ---
# if LOGO_PATH.exists():
#     st.sidebar.image(str(LOGO_PATH), use_container_width=True)

forecasts_df, actuals_df = load_data(FORECAST_PATH, ACTUALS_PATH)

if forecasts_df is not None and actuals_df is not None:
    # --- Sidebar with CASCADING Multi-Select Filters ---
    with st.sidebar:
        st.header("Forecast Filters")


        # Helper for creating multiselect with "All" option
        def multiselect_with_all(label, options, default_all=True):
            all_key = f"All {label}"
            options_with_all = [all_key] + options
            default = [all_key] if default_all else []
            selected = st.multiselect(label, options_with_all, default=default)
            if all_key in selected or not selected:
                return options
            return selected


        # --- Sequential Filtering ---
        # 1. Model
        model_options = sorted(forecasts_df['model'].unique())
        selected_models = multiselect_with_all('Model', model_options)
        df1 = forecasts_df[forecasts_df['model'].isin(selected_models)]

        # 2. Target Variable
        target_options = sorted(df1['target_col'].unique())
        selected_targets = multiselect_with_all('Target Variable', target_options)
        df2 = df1[df1['target_col'].isin(selected_targets)]

        # 3. Horizon
        horizon_options = sorted(df2['horizon'].unique())
        selected_horizons = multiselect_with_all('Horizon (Months)', horizon_options)
        df3 = df2[df2['horizon'].isin(selected_horizons)]

        # 4. Forecast Start Date
        date_options = sorted(df3['forecast_start_date_str'].unique(), reverse=True)
        selected_start_dates = multiselect_with_all('Forecast Start Date', date_options)
        df4 = df3[df3['forecast_start_date_str'].isin(selected_start_dates)]

        st.subheader("Model-Specific Filters")

        # 5. MA Window Filter (enabled only if 'ma' is selected)
        is_ma_selected = 'ma' in selected_models
        window_options = sorted(df4[df4['model'] == 'ma']['window'].unique())
        selected_windows = st.multiselect('MA Window', window_options, default=window_options,
                                          disabled=not is_ma_selected)

        # 6. Prophet Regressor Filter (enabled only if 'prophet' is selected)
        is_prophet_selected = 'prophet' in selected_models
        regressor_options = sorted(df4[df4['model'] == 'prophet']['regressor_str'].unique())
        selected_regressors = st.multiselect('Prophet Regressors', regressor_options, default=regressor_options,
                                             disabled=not is_prophet_selected)

        # --- Final Combination of Filters ---
        ma_df = df4[
            (df4['model'] == 'ma') & (df4['window'].isin(selected_windows))] if is_ma_selected else pd.DataFrame()
        prophet_df = df4[(df4['model'] == 'prophet') & (
            df4['regressor_str'].isin(selected_regressors))] if is_prophet_selected else pd.DataFrame()
        final_filtered_df = pd.concat([ma_df, prophet_df]).sort_index()

    # --- Main Panel Displaying Results ---
    st.header("Filtered Forecast Results")
    st.markdown(f"**Displaying {len(final_filtered_df)} forecast runs.** Use the sidebar to refine your selection.")

    if final_filtered_df.empty:
        st.warning("No forecast runs match the selected criteria. Please adjust sidebar filters.")
    else:
        for _, run_data in final_filtered_df.iterrows():
            if run_data['model'] == 'ma':
                param_str = f"Window: {int(float(run_data['window']))}"
            else:
                param_str = f"Regressors: {run_data['regressor_str']}"

            expander_title = f"**{run_data['model'].upper()}** for **{run_data['target_col']}** | {param_str} | Horizon: {run_data['horizon']}m | Start: {run_data['forecast_start_date_str']}"

            with st.expander(expander_title, expanded=True):
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = plot_forecast(actuals_df, run_data, run_data['target_col'])
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{run_data['unique_key']}")

                with col2:
                    st.subheader("Metrics")
                    st.metric("MAPE", f"{run_data.get('MAPE', 0):.2%}")
                    st.metric("RMSE", f"{run_data.get('RMSE', 0):,.2f}")
                    st.metric("MAE", f"{run_data.get('MAE', 0):,.2f}")
                    st.metric("R² Score", f"{run_data.get('R2', 0):.3f}")
                    st.metric("Bias", f"{run_data.get('Bias', 0):,.2f}")

                    if run_data['model'] == 'prophet' and pd.notna(run_data['best_params']):
                        st.subheader("Best Parameters")
                        try:
                            params_dict = ast.literal_eval(run_data['best_params'])
                            st.json(params_dict, expanded=False)
                        except:
                            st.text(run_data['best_params'])
else:
    st.error("Dashboard could not be loaded. Please check file paths and data format.")
