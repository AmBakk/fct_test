import pandas as pd


def aggregate_timeseries(
        df,
        value_col,
        date_col='Month',
        split=False,
        split_col=None
):
    """Aggregate time series by month, with optional additional split.

    Args:
        df (pd.DataFrame): Input data.
        value_col (str): Name of value column to aggregate.
        date_col (str): Name of date column.
        split (bool): Whether to group by an extra dimension.
        split_col (str): Column name for extra split.

    Returns:
        pd.DataFrame: Aggregated series with all months present, NA filled as 0.
    """
    # Ensure date is parsed
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    group_cols = [date_col]
    if split:
        group_cols.append(split_col)

    # Aggregate
    agg = df.groupby(group_cols)[value_col].sum().reset_index()

    # Build complete date index
    date_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq='MS')

    if split:
        out = []
        for nat_val in agg[split_col].unique():
            temp = agg[agg[split_col] == nat_val].set_index(date_col)
            temp = temp.reindex(date_range, fill_value=0)
            temp[split_col] = nat_val
            temp = temp.reset_index().rename(columns={'index': date_col})
            out.append(temp)
        agg_full = pd.concat(out, ignore_index=True)
    else:
        agg_full = agg.set_index(date_col).reindex(date_range, fill_value=0)
        agg_full = agg_full.reset_index().rename(columns={'index': date_col})

    return agg_full


