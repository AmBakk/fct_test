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
# Navigate to the project root (assuming the script is in src/utils)
# Adjust this if your structure is different.
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Construct absolute paths
# --- IMPORTANT: UPDATE FILENAMES HERE ---
FORECAST_PATH = PROJECT_ROOT / "Data/processed/Forecasts/forecast_summary_20250708_031202.csv"
ACTUALS_PATH = PROJECT_ROOT / "Data/processed/final_merge.csv"


# ------------------------------------

# --- Data Loading and Caching ---
@st.cache_data
def load_data(forecast_path, actuals_path):
    """
    Loads and preprocesses forecast and actuals data.
    This function is cached to improve performance.
    """
    # --- Load Forecast Data ---
    try:
        forecasts_df = pd.read_csv(forecast_path)
    except FileNotFoundError:
        st.error(f"Forecast file not found. Checked path: {forecast_path}")
        return None, None

    # --- Preprocess Forecasts DF ---
    metric_cols = ['MAE', 'MAPE', 'RMSE', 'R2', 'Bias']
    for col in metric_cols:
        if col in forecasts_df.columns:
            forecasts_df[col] = forecasts_df[col].astype(str)
            forecasts_df[col] = pd.to_numeric(
                forecasts_df[col].str.replace('%', '', regex=False),
                errors='coerce'
            )
            if col == 'MAPE' and any(forecasts_df[col] > 1):
                forecasts_df[col] = forecasts_df[col] / 100.0

    # Handle multiple date formats gracefully
    forecasts_df['end_date'] = pd.to_datetime(forecasts_df['end_date'], errors='coerce')
    forecasts_df.sort_values('end_date', ascending=False, inplace=True)
    forecasts_df['window'] = forecasts_df['window'].fillna('N/A')

    def create_regressor_str(row):
        if not row['use_regressors']:
            return 'No Regressors'
        try:
            cols = ast.literal_eval(row['regressor_cols'])
            return ', '.join(cols)
        except (ValueError, SyntaxError, TypeError):
            return str(row['regressor_cols'])

    forecasts_df['regressor_str'] = forecasts_df.apply(create_regressor_str, axis=1)

    # --- ROBUST PARSING FOR 'forecast_json' ---
    def parse_forecast_json(json_like_str):
        if pd.isna(json_like_str):
            return pd.DataFrame()
        try:
            # Step 1: Replace Timestamp object with a parsable string
            # This regex finds "Timestamp('...') and extracts the date string inside
            processed_str = re.sub(r"Timestamp\('([^']*)'\)", r"'\1'", json_like_str)
            # Step 2: Safely evaluate the string to a Python list of dicts
            data = ast.literal_eval(processed_str)
            # Step 3: Convert to DataFrame
            df = pd.DataFrame(data)
            df['Month'] = pd.to_datetime(df['Month'])
            return df
        except (ValueError, SyntaxError, TypeError) as e:
            # st.warning(f"Could not parse forecast_json: {e}\nString was: {json_like_str[:100]}...")
            return pd.DataFrame()

    forecasts_df['forecast_df'] = forecasts_df['forecast_json'].apply(parse_forecast_json)

    # --- Load and Aggregate Actuals DF ---
    try:
        actuals_source_df = pd.read_csv(actuals_path)
    except FileNotFoundError:
        st.error(f"Actuals file not found at: {actuals_path}")
        return None, None

    actuals_source_df['Month'] = pd.to_datetime(actuals_source_df['Month'])

    # Correctly aggregate the data to the monthly level
    actuals_df = actuals_source_df.groupby('Month')[['Weight Sold', 'Weight Consumed']].sum().reset_index()
    actuals_df.set_index('Month', inplace=True)

    return forecasts_df, actuals_df


# --- Plotting Function (No changes needed) ---
def plot_forecast(actuals_df, forecast_run, target_col):
    """
    Generates an interactive Plotly chart for a single forecast run
    with improved visual continuity and accurate labeling.
    """
    fig = go.Figure()

    forecast_df = forecast_run['forecast_df'].copy()
    if forecast_df.empty:
        # Cannot plot if there's no forecast data
        return fig

        # --- Fix #2: Determine the correct forecast start and end dates ---
    forecast_start_date = forecast_df['Month'].min()
    forecast_end_date = forecast_df['Month'].max()

    # --- Fix #3: Filter actuals to not show data beyond the forecast horizon ---
    plot_actuals_df = actuals_df[actuals_df.index <= forecast_end_date]

    # 1. Plot the historical and relevant future actuals
    fig.add_trace(go.Scatter(
        x=plot_actuals_df.index,
        y=plot_actuals_df[target_col],
        mode='lines',
        name='Actuals (Full History)',
        line=dict(color='gray')
    ))

    # --- Fix #1: Create a continuous forecast line ---
    # Get the last actual data point before the forecast starts
    last_actual_date = forecast_start_date - pd.DateOffset(months=1)
    last_actual_row = actuals_df.loc[[last_actual_date]].reset_index()

    # Create the continuous forecast series
    continuous_forecast_df = pd.concat([
        pd.DataFrame([{
            'Month': last_actual_date,
            'forecast': last_actual_row.iloc[0][target_col]
        }]),
        forecast_df[['Month', 'forecast']]
    ], ignore_index=True)

    # 2. Plot the continuous forecast line
    fig.add_trace(go.Scatter(
        x=continuous_forecast_df['Month'],
        y=continuous_forecast_df['forecast'],
        mode='lines',
        name=f"Forecast ({forecast_run['model']})",
        line=dict(color='red', dash='dash')
    ))

    # 3. Plot the actuals during the forecast period for comparison
    fig.add_trace(go.Scatter(
        x=forecast_df['Month'],
        y=forecast_df[target_col],
        mode='lines',
        name='Actuals (During Forecast)',
        line=dict(color='blue', width=2.5)
    ))

    # 4. Add a vertical line at the CORRECT forecast start date
    fig.add_vline(
        x=forecast_start_date,  # Use the corrected start date
        line_width=2,
        line_dash="dot",
        line_color="black"
    )

    # 5. Add the annotation manually
    fig.add_annotation(
        x=forecast_start_date,
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=True,
        arrowhead=7,
        ax=0,
        ay=-40
    )

    fig.update_layout(
        title=f"Forecast vs. Actuals for: {forecast_run['run_id']}",
        xaxis_title="Month",
        yaxis_title=target_col,
        legend_title="Legend"
    )

    return fig


# --- Main App Logic ---
forecasts_df, actuals_df = load_data(FORECAST_PATH, ACTUALS_PATH)

if forecasts_df is not None and actuals_df is not None:
    # --- Sidebar Filters ---
    with st.sidebar:
        st.header("1. Select Primary Filters")

        selected_target = st.selectbox(
            'Target Variable',
            options=forecasts_df['target_col'].unique()
        )

        selected_model = st.selectbox(
            'Forecasting Model',
            options=forecasts_df['model'].unique()
        )

        # Filter dataframe based on primary selections
        filtered_df = forecasts_df[
            (forecasts_df['target_col'] == selected_target) &
            (forecasts_df['model'] == selected_model)
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning

        st.header("2. Refine Selections")

        selected_horizon = st.selectbox(
            'Forecast Horizon (Months)',
            options=sorted(filtered_df['horizon'].unique())
        )

        filtered_df = filtered_df[filtered_df['horizon'] == selected_horizon]

        if selected_model == 'ma':
            selected_window = st.selectbox(
                'Moving Average Window',
                options=sorted(filtered_df['window'].unique())
            )
            filtered_df = filtered_df[filtered_df['window'] == selected_window]

        elif selected_model == 'prophet':
            regressor_options = sorted(filtered_df['regressor_str'].unique())
            if regressor_options:
                selected_regressor = st.selectbox(
                    'Regressor(s) Used',
                    options=regressor_options
                )
                filtered_df = filtered_df[filtered_df['regressor_str'] == selected_regressor]

        date_options = sorted(filtered_df['end_date'].dt.date.unique(), reverse=True)
        if date_options:
            selected_end_date = st.selectbox(
                'Backtest End Date',
                options=date_options
            )
            final_filtered_df = filtered_df[filtered_df['end_date'].dt.date == selected_end_date]
        else:
            final_filtered_df = pd.DataFrame()  # No dates match, so empty df

    # --- Main Panel for Comparison ---
    st.header("3. Compare Forecast Runs")

    if final_filtered_df.empty:
        st.warning("No forecast runs match the selected criteria. Please adjust your filters.")
    else:
        selected_runs = st.multiselect(
            'Select runs to display:',
            options=final_filtered_df['run_id'].tolist(),
            default=final_filtered_df['run_id'].tolist()[:1]
        )

        if not selected_runs:
            st.info("Select at least one run from the dropdown above to see the results.")
        else:
            for run_id in selected_runs:
                run_data = final_filtered_df[final_filtered_df['run_id'] == run_id].iloc[0]

                with st.expander(f"**Run ID:** {run_id}", expanded=True):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.subheader("Metrics")
                        mape_val = run_data['MAPE']
                        if pd.notna(mape_val):
                            st.metric("MAPE", f"{mape_val:.2%}")
                        st.metric("RMSE", f"{run_data['RMSE']:,.2f}")
                        st.metric("MAE", f"{run_data['MAE']:,.2f}")
                        st.metric("R² Score", f"{run_data['R2']:.3f}")
                        st.metric("Bias", f"{run_data['Bias']:,.2f}")

                        if run_data['model'] == 'prophet' and pd.notna(run_data['best_params']):
                            st.subheader("Best Parameters")
                            try:
                                params_dict = ast.literal_eval(run_data['best_params'])
                                st.json(params_dict, expanded=False)
                            except (ValueError, SyntaxError):
                                st.text(run_data['best_params'])

                    with col2:
                        st.subheader("Forecast Plot")
                        if not run_data['forecast_df'].empty:
                            fig = plot_forecast(actuals_df, run_data, selected_target)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(
                                "Could not generate plot for this run (forecast data might be missing or failed to parse).")
else:
    st.error(
        "Dashboard could not be loaded. Please check the file paths and ensure the CSVs are in the correct format.")