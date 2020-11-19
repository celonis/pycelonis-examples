from pycelonis import pql

import datetime
import isoweek
import pandas as pd

## Loading Data


def get_pql_dataframe(dm, input_columns, input_filter):
    """Query input columns with filters from input DM"""
    query = pql.PQL()
    for col_name, col_pretty_name in input_columns:
        query += pql.PQLColumn(col_name, col_pretty_name)
    if input_filter != '':
        query += pql.PQLFilter(input_filter)
    queried_df = dm.get_data_frame(query)
    return queried_df


def get_subset_df(train_df, subset, subset_col_name):
    """Filter df for subset"""
    subset_train_df = train_df[train_df[subset_col_name] == subset]
    subset_train_df.drop(columns=[subset_col_name], inplace=True)
    return subset_train_df


## Pre-processing


def fill_empty_dates(df):
    """Fill empty weeks of date Df"""
    my_date = datetime.datetime.now()
    year, week_num, day_of_week = my_date.isocalendar()
    d = isoweek.Week(year, week_num - 1).monday()
    rng = pd.date_range(df["Date"].min(), d, freq="7D")
    df = df.set_index("Date").reindex(rng, fill_value=0).reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)
    return df


def cap_outliers(df, max_outlier_value):
    """Clean outliers"""
    df.loc[df["Net Order Value"] > max_outlier_value,
           "Net Order Value"] = max_outlier_value
    return df


def adjust_baseline(df, change_date, end_date):
    """Calculate baseline avg difference between TS before change_date vs TS between change_date and end_date"""
    diff_high_low = (
        df.loc[(change_date < df["Date"]) &
               (df["Date"] <= end_date), "Net Order Value"].mean() -
        df.loc[df["Date"] <= change_date, "Net Order Value"].mean())
    # Adjust lower baseline with the above avg difference
    df.loc[df["Date"] <= change_date, "Net Order Value"] += diff_high_low
    return df


## Model utils


def calculate_trend(df, ts_seasonality, center=False):
    """Calculate Trend"""
    t = df.iloc[:, 1].rolling(window=ts_seasonality, center=center).mean()
    return t


def combine_ext_data(train_df, ext_data, days_to_shift=None):
    """Combine External/GDP data with Y"""
    # Add Exo regressors (GDP) to train df
    train_df = train_df.set_index("Date")
    ext_data["DATE"] = pd.to_datetime(ext_data["DATE"])
    ext_data = ext_data.set_index("DATE")
    # Optional - Align dates of Industry GDP with Trend
    if days_to_shift is not None:
        ext_data = ext_data.shift(days_to_shift, freq="D")
    # Combine Train Df with GDP
    train_df = train_df.combine_first(ext_data)
    return train_df


def get_trend_and_exo_for_fit(train_df, exo_col_name, trend_col_name,
                              val_size_perc):
    """Create subsets for Trend Fit"""
    # Create X set (Exo Regressor)
    X = train_df.dropna()[exo_col_name].values
    train_size = int(len(X) * (1 - val_size_perc))
    X_train = X[:train_size].reshape(-1, 1)
    # Create Y set (Trend to fit)
    Y_train = train_df.dropna()[trend_col_name].values[:train_size].reshape(
        -1, 1)
    return X_train, Y_train


def fill_seasonality(train_df,
                     seas_period_days,
                     seasonality_col_name='Seasonality'):
    """Fill empty seasonality dates"""
    delta = datetime.timedelta(days=-seas_period_days)
    for i in train_df[train_df[seasonality_col_name].isnull() == True].index:
        print(i, i + delta)
        train_df.loc[i][seasonality_col_name] = train_df.loc[
            i + delta][seasonality_col_name]
    return train_df


## Exports


def prepare_export_df(train_df, output_col_names, y_pred_col_name):
    """Reformat results for Export to DM"""
    print(output_col_names)
    cols_to_load = list(output_col_names)
    cols_to_load.remove('index')
    print(cols_to_load)
    export_df = pd.DataFrame(train_df[cols_to_load])
    export_df.reset_index(inplace=True)
    export_df.rename(columns=output_col_names, inplace=True)
    return export_df


def constitute_export_df(all_subset_exports, subset_col_name):
    """Create export-version Df from the export-version of subsets"""
    export_df = pd.DataFrame()
    for key in all_subset_exports:
        subset_df = all_subset_exports[key]
        subset_df[subset_col_name] = key
        export_df = pd.concat([export_df, subset_df], axis=0)
    return export_df