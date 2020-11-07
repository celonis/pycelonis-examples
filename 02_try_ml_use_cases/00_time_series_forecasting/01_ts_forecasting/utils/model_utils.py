from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima
import statsmodels.api as sm
from sklearn import linear_model, metrics
from statsmodels.tsa.statespace import sarimax
from .utils import utils, plot_utils


def run_predictions_model(df, ext_data, val_size_perc, to_adjust_years):
    """Run Predictions Model for Train df

    Parameters
    ----------
    df : [type]
        [description]
    ext_data : [type]
        [description]
    val_size_perc : [type]
        [description]
    to_adjust_years : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Reindex and print Df
    df = df.reset_index()
    # Clean data: fill empty weeks with 0 value
    df = utils.fix_data(df)
    # Create Train Df
    train_df = df.copy()
    # Cap the high outliers to a max value
    # PARAM - Max value
    max_outlier_value = 1000
    train_df = utils.cap_outliers(train_df, max_outlier_value)
    # Adjust past data if baseline changed at date D
    if to_adjust_years:
        # PARAM - dates
        change_date = "TBD"
        end_date = "TBD"
        train_df = utils.adjust_baseline(train_df, change_date, end_date)
    # Plot
    # PARAM - y margin for y axis
    y_margin = 500000
    plot_utils.plot_clean_y(df, train_df, max_outlier_value, y_margin)
    # MODEL: Y = Trend + Seasonality + Residuals
    # Trend: Calculate, Model and Predict future values
    # PARAM - Trend window e.g. 52 if weekly TS with annual seasonality. 7 if daily TS with weekly seasonality
    ts_seasonality = 52
    train_df["Trend"] = utils.calculate_trend(train_df, ts_seasonality, center=False)
    # Plot Y and Trend
    # PARAM - y axis
    y_min = 0
    y_max = 9000000
    plot_utils.plot_y_trend(train_df, t, y_min, y_max)

    # Use External data/GDP to fit and predict the Trend
    print(train_df.dropna().shape)
    train_df = utils.combine_ext_data(train_df, ext_data, days_to_shift=1)
    # PARAM - External Data/GDP column
    exo_col_name = "TBD"
    exo_pretty_name = "TBD"
    # Regression Trend on External data/GDP: define X=GDP and Y=Trend for regression model
    X, Y = utils.subsets_to_fit(train_df, exo_col_name, "Trend", val_size_perc)
    # Plot Y, Trend and Exo Regr
    # PARAM - y axis scale for External data/GDP
    y_min_gdp = 100
    y_max_gdp = 200
    plot_utils.plot_y_trend_ext(train_df, exo_col_name, exo_pretty_name, y_min, y_max, y_min_gdp, y_max_gdp)
    # Fit Regression Y=Trend X=Exo
    reg = linear_model.LinearRegression().fit(X, Y)
    print(reg.coef_)
    print(reg.intercept_)
    print(reg.score(X, Y))
    # Predict Trend with fitted Regression
    trend_pred_col_name = "Predicted Trend"
    X_F, train_df = predict_trend(train_df, exo_col_name, trend_pred_col_name)
    # Plot Trend, External data/GDP and Predicted Trend
    plot_utils.plot_y_pred_trend_ext(train_df, exo_col_name, X, Y, X_F, y_min, y_max, y_min_gdp, y_max_gdp)
    print(train_df.head())

    # Calculate Seasonality
    # Calculate Y - Trend
    train_df["Y - Trend"] = train_df[y_col_name] - train_df["Trend"]
    # Get Seasonality by moving avg on Y - T, and average across years for 1 value per week of year
    # PARAM - Moving avg window for S
    window = 10
    s = train_df["Y - Trend"].rolling(window=window, center=True).mean()
    s = s.groupby(s.index.week).mean()
    print(s.head(5))
    print(s.tail(5))
    # Add Seasonality to df (Assign S to each week of each year)
    # PARAM - S column name
    seasonality_col_name = "Seasonality"
    train_df[seasonality_col_name] = np.nan
    for i in train_df.index:
        train_df.loc[i][seasonality_col_name] = s[i.week]
    # Should be not be required #
    # Fix border dates with Null values
    # PARAM - seasonsality period in days
    seas_period_days = 52 * 7
    delta = datetime.timedelta(days=-seas_period_days)
    for i in train_df[train_df[seasonality_col_name].isnull() == True].index:
        train_df.loc[i][seasonality_col_name] = train_df.loc[i + delta][seasonality_col_name]
    # Plot Y, T and S
    plot_utils.plot_y_t_s_with_pred(train_df, trend_col_name, seasonality_col_name, trend_pred_col_name)
    # Residuals: Calculate, Model and Predict future values
    # Calculate R = Y - Trend - Season
    train_df["Y - T - S"] = train_df[y_col_name] - train_df[trend_col_name] - train_df[seasonality_col_name]
    # Create R df
    # PARAM - R column name
    r_col_name = "Y - T - S"
    r = train_df[r_col_name]
    # Plot R
    plot_utils.plot_r(train_df, r_col_name)
    # R Study
    # R shape
    print(r.dropna().shape)
    # Stationarity test
    res = sm.tsa.adfuller(r.dropna(), regression="c")
    print("p-value:{}".format(res[1]))
    # Verify that p value is low
    # ACF PACF on R
    # PARAM - # lags for acf pacf
    lags = 25
    plot_utils.plot_acf_pacf_r(r, lags)
    # Deduce ARMA(p,q) model for R

    # Create R df for R Model
    columns_to_drop = [y_col_name, exo_col_name]
    col_to_rename = {"index": "Date"}
    r_df = create_r_df(train_df, columns_to_drop, col_to_rename)

    # Fit ARIMA Model on R for R predictions
    # PARAM - p for AR, d for I, q for MA. Set using acf pacf plots above.
    # P,D,Q,s can remain None.
    # n_pred is # future points to forecast
    # (Optional) model - to input an existing loaded model
    # (Optional) exo - to input exogenous regressors
    p, d, q = 3, 0, 3
    P, D, Q, s = None, None, None, None
    n_pred = 18
    model = None
    exo = None
    r_df = r_df.dropna()
    model_r, results_df_r = get_results_with_val(
        r_df, exo, p, d, q, P, D, Q, s, model, value_col_name, val_size_perc, n_pred
    )
    # Add Predicted R to df
    # PARAM - R column name for df
    r_col_name = "Predicted R"
    r_class_col_name = "Predicted R Classification"
    train_df = add_r(train_df, results_df_r, r_col_name, r_class_col_name)

    # Calculate Total Y Prediction = Predicted T + S + Predicted R
    # PARAM - y pred column names
    y_pred_col_name = "Y Prediction"
    train_df = calc_y_pred(train_df, y_pred_col_name, trend_pred_col_name, seasonality_col_name, r_class_col_name)
    print(train_df.tail(n=20))
    # Plot and show Final Df with predictions
    plot_utils.plot_final_y_t_s_r_with_pred(
        train_df, trend_col_name, seasonality_col_name, r_col_name, trend_pred_col_name, y_pred_col_name
    )
    # Show Final Df
    train_df.head(n=10)
    train_df.tail(n=10)
    return train_df


def get_results_with_val(df, exo, p, d, q, P, D, Q, s, model, value_col_name, val_size_perc, n_predictions=18):
    """Fit ARIMA on input df (optional input and future exo regr) and predict validation + future values
    Or use param fitted model (optional input and future exo regr) to predict validation + future values
    Plot input and output (val+future) predictions

    Parameters
    ----------
    df : [type]
        [description]
    exo : [type]
        [description]
    p : [type]
        [description]
    d : [type]
        [description]
    q : [type]
        [description]
    P : [type]
        [description]
    D : [type]
        [description]
    Q : [type]
        [description]
    s : [type]
        [description]
    model : [type]
        [description]
    value_col_name : [type]
        [description]
    val_size_perc : [type]
        [description]
    n_predictions : int, optional
        [description], by default 18

    Returns
    -------
    [type]
        [description]
    """

    X = df[value_col_name].values
    Y = df["Date"].values
    train_size = int(len(X) * (1 - val_size_perc))
    train, test = X[:train_size], X[train_size : len(X)]
    week = Y[train_size : len(X)]
    exo_past, exo_future = None, None
    if exo is not None:
        exo_past, exo_future = exo[: len(X)], exo[len(X) : len(exo)]
    print("Checking model for fit...")
    if model is None:
        print("No input model, will fit SARIMA" + str(p) + str(d) + str(q) + str(P) + str(D) + str(Q) + str(s))
        print("Starting Arima fit...")
        smodel = pmdarima.arima.ARIMA(order=[p, d, q], method="lbfgs", maxiter=50, suppress_warnings=True)
        smodel = smodel.fit(df[value_col_name].values, exo_past)
        print("Finished Arima fit.")
    else:
        print("Existing model, will use it")
        smodel = model

    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = sarimax.SARIMAX(
            history, order=smodel.order, seasonal_order=smodel.seasonal_order, enforce_stationarity=False
        )
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        if output[0] < 0:
            yhat = 0
        else:
            yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print("predicted=%f, expected=%f" % (yhat, obs))
    error = metrics.mean_squared_error(test, predictions)
    print("Test MSE: %.3f" % error)

    # Train set (train data)
    data = pd.DataFrame()
    data["Date"] = Y[0:train_size]
    data["Predicted Net Order Value"] = None
    data["Actual Net Order Value"] = X[0:train_size]
    data["Classification"] = "train"

    # Validation set (val data with predictions)
    Tested = pd.DataFrame()
    Tested["Date"] = week
    Tested["Predicted Net Order Value"] = predictions
    Tested["Actual Net Order Value"] = test
    Tested["Classification"] = "test"
    Tested["Predicted Net Order Value"] = Tested["Predicted Net Order Value"].astype(float)
    Tested["Date"] = pd.to_datetime(Tested["Date"])

    # Forecast set (out-of-sample predictions)
    print("Starting to predict future values...")
    n_periods = n_predictions
    fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True, exogenous=exo_future)
    print("Finished to predict future values.")
    rng = pd.date_range(df["Date"].max(), periods=n_periods, freq="7D")
    forecast = pd.DataFrame(
        {
            "Date": rng,
            "Predicted Net Order Value": fitted,
            "Actual Net Order Value": None,
            "Classification": "forecast",
            "Conf_lower": confint[:, 0],
            "Conf_Upper": confint[:, 1],
        }
    )
    forecast = forecast.drop(forecast.index[0])

    # All sets combined
    results = data.append(Tested, ignore_index=True)
    results = results.append(forecast, ignore_index=True)
    results["Date"] = pd.to_datetime(results["Date"])
    # Reformat Dates to Date type
    results["Date"] = pd.to_datetime(results["Date"])
    return smodel, results


def calc_y_pred(train_df, y_pred_col_name, trend_pred_col_name, seasonality_col_name, r_class_col_name):
    train_df[y_pred_col_name] = np.nan
    # Validation Y values
    mask = train_df[r_class_col_name] == "test"
    train_df.loc[mask, y_pred_col_name] = (
        train_df[trend_pred_col_name] + train_df[seasonality_col_name] + train_df["Predicted R"]
    )
    # Future Y values
    mask = train_df[r_class_col_name] == "forecast"
    train_df.loc[mask, y_pred_col_name] = (
        train_df[trend_pred_col_name] + train_df[seasonality_col_name] + train_df["Predicted R"]
    )


def predict_trend(train_df, exo_col_name, pred_trend_col_name):
    """Trend Regression to predict future Trend"""
    X_F = train_df[exo_col_name].dropna().values.reshape(-1, 1)
    print(X_F.shape)
    print(reg.predict(X_F).shape)
    # Add Predicted Trend to df
    t_pred = reg.predict(X_F)
    len_pred = t_pred.shape[0]
    train_df["Predicted Trend"] = np.nan
    train_df["Predicted Trend"][-len_pred:] = t_pred.ravel()
    return X_F, train_df


def create_r_df(train_df, columns_to_drop, col_to_rename):
    """Residuals"""
    r_df = train_df.copy()
    r_df = r_df.drop(columns=columns_to_drop)
    r_df = r_df.reset_index()
    r_df = r_df.rename(columns=col_to_rename)
    return r_df


def add_r(train_df, results_df_r, r_col_name):
    results_df_r_idx = results_df_r.set_index("Date")
    train_df[r_col_name] = np.nan
    train_df[r_col_name] = results_df_r_idx["Predicted Net Order Value"]
    train_df[r_class_col_name] = results_df_r_idx["Classification"]
    return train_df
