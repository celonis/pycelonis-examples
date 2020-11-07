import datetime

import isoweek
import pandas as pd


def summarize_df(df):
    """Summarize df with Date"""
    print(df.shape[0])
    print(min(df["Date"]))
    print(max(df["Date"]))
    df


def prepare_export_df(train_df, output_col_names, y_pred_col_name):
    """Reformat results for Export to DM"""
    export_df = pd.DataFrame(train_df[[y_col_name, y_pred_col_name, r_class_col_name]])
    export_df.reset_index(inplace=True)
    export_df.rename(columns=output_col_names, inplace=True)
    print(export_df.shape)
    return export_df


def fix_data(df):
    """Fill empty w#eeks of date Df"""
    my_date = datetime.datetime.now()
    year, week_num, day_of_week = my_date.isocalendar()
    d = isoweek.Week(year, week_num - 1).monday()
    rng = pd.date_range(df["Date"].min(), d, freq="7D")
    df = df.set_index("Date").reindex(rng, fill_value=0).reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)
    return df


def cap_outliers(df, max_outlier_value):
    """Clean outliers"""
    df.loc[df["Net Order Value"] > max_outlier_value, "Net Order Value"] = max_outlier_value
    return df


def adjust_baseline(df, change_date, end_date):
    """Calculate baseline avg difference between TS before change_date vs TS between change_date and end_date"""
    diff_high_low = (
        df.loc[(change_date < df["Date"]) & (df["Date"] <= end_date), "Net Order Value"].mean()
        - df.loc[df["Date"] <= change_date, "Net Order Value"].mean()
    )
    # Adjust lower baseline with the above avg difference
    df.loc[df["Date"] <= change_date, "Net Order Value"] += diff_high_low
    return df


def calculate_trend(df, ts_seasonality, center=False):
    t = df.iloc[:, 1].rolling(window=ts_seasonality, center=center).mean()
    return t


def combine_ext_data(train_df, ext_data, days_to_shift=1):
    """Combine External/GDP data with Y"""
    # Add Exo regressors (GDP) to train df
    train_df = train_df.set_index("Date")
    ext_data["DATE"] = pd.to_datetime(ext_data["DATE"])
    # Optional - Align dates of Industry GDP with Trend
    if days_to_shift is not None:
        ext_data = ext_data.set_index("DATE").shift(days_to_shift, freq="D")
    # Combine Train Df with GDP
    train_df = train_df.combine_first(ext_data)
    return train_df


def subsets_to_fit(train_df, exo_col_name, trend_col_name, val_size_perc):
    """Create subsets for Trend Fit"""
    # Create X set (exo regressor)
    X = train_df.dropna()[exo_col_name].values
    train_size = int(len(X) * (1 - val_size_perc))
    X_train = X[:train_size].reshape(-1, 1)
    # Create Y set (Trend to fit)
    Y_train = train_df.dropna()[trend_col_name].values[:train_size].reshape(-1, 1)
    return X_train, Y_train
