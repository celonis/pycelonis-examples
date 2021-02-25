from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pmdarima
import statsmodels.api as sm
from sklearn import linear_model, metrics
from statsmodels.tsa.statespace import sarimax
from . import utils, plot_utils


def run_predictions_model(df,
                          ext_data,
                          y_col_name,
                          exo_col_name,
                          val_size_perc=0.2,
                          to_adjust_years=False):
    """Run Predictions Model for Train df

    Parameters
    ----------
    df : DataFrame
        Train set Dataframe containing the Y values of the Time Series to predict
    ext_data : DataFrame
        External data to use as Regressor to model and predict the TS Trend
    val_size_perc : Float
        Part of the df to use for Validation. 
        Format: [0.0;1.0]
    to_adjust_years : Boolean
        True if baseline level of the TS has changed during its timeframe and should be adjusted
        By default False
    y_col_name : String
        Column name of the TS values column (Y column).
    exo_col_name : String
        Column name of the External Regressor values column.

    Returns
    -------
    DataFrame
        Output DataFrame with the n-step Predictions for the TS (Predict the n future Y values).
        n is set as the minimum between the number of future values from the External data and the predicted Residuals
    """

    # Reindex and create Train Df
    df = df.reset_index(drop=True)
    train_df = df.copy()
    print('train df head looks like: \n', train_df.head())

    # Clean data: fill empty weeks with 0 value
    train_df = utils.fill_empty_dates(train_df)

    # Cap the high outliers to a max value
    train_df = utils.cap_outliers(
        train_df,
        max_outlier_value=1000)  # PARAM - max_outlier_value: Max value

    # Adjust past data if baseline changed at date change_date
    if to_adjust_years:
        train_df = utils.adjust_baseline(train_df,
                                         change_date='YYYY-MM-DD',
                                         end_date='YYYY-MM-DD')
        # PARAM - change_date: date at which baseline level changed, end_date: end date of new baseline level

    # Plot preprocessed Train Df
    plot_utils.plot_clean_y(df, train_df,
                            y_max=1000 + 100)  #PARAM - y axis max value

    #### MODEL: Y = Trend + Seasonality + Residuals

    ### Trend: Calculate, Model and Predict future values
    trend_col_name = 'Trend'  # PARAM - Trend column name
    train_df[trend_col_name] = utils.calculate_trend(
        train_df,
        ts_seasonality=
        52,  # PARAM - Seasonality timeframe e.g. 52 if weekly data with annual seasonality. 7 if daily TS with weekly seasonality
        center=False)
    # Plot Y and Trend
    plot_utils.plot_y_trend(train_df,
                            train_df[trend_col_name],
                            y_min=0,
                            y_max=100)

    # Use External data/GDP to fit and predict the Trend
    print('train df shape is ',
          train_df.dropna().shape, ', adding the external data into the df...')
    train_df = utils.combine_ext_data(train_df, ext_data, days_to_shift=None)

    # Define X=GDP and Y=Trend for Regression model
    exo_pretty_name = "Regressor"  # PARAM - External Data/GDP column
    X, Y = utils.get_trend_and_exo_for_fit(train_df, exo_col_name,
                                           trend_col_name, val_size_perc)
    # Plot Y, Trend and Exo Regr
    plot_utils.plot_y_trend_ext(train_df,
                                Y,
                                exo_col_name,
                                exo_pretty_name,
                                y_min=0,
                                y_max=1100,
                                y_min_exo=100,
                                y_max_exo=200)

    # Fit Regression of Y=Trend on X=Exogenous Regressor
    reg = linear_model.LinearRegression().fit(X, Y)
    # Predict future Trend with the fitted Regression
    trend_pred_col_name = "Predicted Trend"
    X_F, train_df = predict_trend(train_df, reg, exo_col_name,
                                  trend_pred_col_name)
    # Plot Trend, External data/GDP and Predicted Trend
    plot_utils.plot_y_pred_trend_ext(train_df,
                                     exo_col_name,
                                     X,
                                     Y,
                                     X_F,
                                     y_min=0,
                                     y_max=1100,
                                     y_min_exo=100,
                                     y_max_exo=200)
    print('End of Trend part, df is \n', train_df.head())

    ### Seasonality: Calculate S for each date of the seasonality window

    # Calculate Y - Trend
    train_df["Y - Trend"] = train_df[y_col_name] - train_df[trend_col_name]

    # Calculate Seasonality by moving avg on Y - T
    s = train_df["Y - Trend"].rolling(
        window=10,
        center=True).mean()  # PARAM - window: Moving avg window to smoothen S
    # Avg across periods to obtain 1 S value per date of a period
    s = s.groupby(s.index.week).mean()

    # Add Seasonality to Df
    seasonality_col_name = "Seasonality"  # PARAM - S column name
    train_df[seasonality_col_name] = np.nan
    for i in train_df.index:
        train_df.loc[i][seasonality_col_name] = s[i.week]
    # (Optional) Fix border dates with Null values
    # seas_period_days = 52 * 7  # PARAM - seasonsality period in days
    # train_df = utils.fill_seasonality(train_df, seas_period_days,
    #                                  seasonality_col_name)

    # Plot Y, T and S
    plot_utils.plot_y_t_s_with_pred(train_df, trend_col_name,
                                    seasonality_col_name, trend_pred_col_name)

    ### Residuals: Calculate, Model and Predict future values

    # Calculate R = Y - Trend - Season
    train_df["Y - T - S"] = train_df[y_col_name] - train_df[
        trend_col_name] - train_df[seasonality_col_name]
    # Create R df
    r_col_name = "Y - T - S"  # PARAM - R column name
    r = train_df[r_col_name]
    # Plot R
    plot_utils.plot_r(train_df, r_col_name)
    # R shape
    print('R df shape is ', r.dropna().shape)
    # Stationarity test
    res = sm.tsa.adfuller(r.dropna(), regression="c")
    print("adf test p-value is:{}".format(res[1]))
    # Verify that p value is low
    # ACF PACF on R
    plot_utils.plot_acf_pacf_r(r, lags=25)  # PARAM - # lags for acf pacf
    # Deduce ARMA(p,q) model for R

    # Create R df for R Model
    columns_to_drop = [y_col_name, exo_col_name]
    col_to_rename = {"index": "Date"}
    r_df = create_r_df(train_df, columns_to_drop, col_to_rename)

    # Fit ARIMA Model on R for R predictions
    p, d, q = 3, 0, 3  # PARAM - p for AR, d for I, q for MA.
    P, D, Q, s = None, None, None, None  # If seasonality use P,D,Q,s, if not set to None.
    n_pred = 5  # n_pred is # future points to forecast
    model = None  # (Optional) model - to input an existing loaded model
    exo = None  # (Optional) exo - to input exogenous regressors
    r_df = r_df.dropna()
    model_r, results_df_r = get_results_with_val(r_df, exo, p, d, q, P, D, Q, s,
                                                 model, r_col_name,
                                                 val_size_perc, n_pred)
    # Add Predicted R to df
    r_col_name = "Predicted R"  # PARAM - R column name for df
    class_col_name = "Classification"  # PARAM - classification col name (train/test/forecast)
    train_df = add_r(train_df, results_df_r, r_col_name, class_col_name)

    ### Calculate Total Y Prediction = Predicted T + S + Predicted R

    y_pred_col_name = "Y Prediction"  # PARAM - y pred column names
    train_df = calc_y_pred(train_df, y_pred_col_name, trend_pred_col_name,
                           seasonality_col_name, class_col_name)
    print('End of df with predictions is \n', train_df.tail(n=20))

    # Plot and show Final Df with predictions
    plot_utils.plot_final(train_df, trend_col_name, seasonality_col_name,
                          r_col_name, trend_pred_col_name, y_pred_col_name,
                          class_col_name)

    # Return Final Df with Y predictions
    return train_df


def get_results_with_val(df,
                         exo,
                         p,
                         d,
                         q,
                         P,
                         D,
                         Q,
                         s,
                         model,
                         y_col_name,
                         val_size_perc,
                         n_predictions=5):
    """Fit SARIMAX on input df (optional input and future exo regr) and predict validation + future values
    Or use param fitted model (optional input and future exo regr) to predict validation + future values
    Plot input and output (val+future) predictions

    Parameters
    ----------
    df : DataFrame
        R Time Series
    exo : DataFrame, optional
        Exogenous Regressors to model Y
    p : int
        AR parameter for the SARIMAX on Y
    d : int
        Integrated parameter for the SARIMAX on Y
    q : int
        MA parameter for the SARIMAX on Y
    P : int
        Seasonal AR parameter for the SARIMAX on Y
    D : int
        Seasonal Integrated parameter for the SARIMAX on Y
    Q : int
        Seasonal MA parameter for the SARIMAX on Y
    s : int
        Seasonality timeframe for Y
    model : SARIMAX Fitted model, optional
        Pre-fitted SARIMAX model to use to predict Y values
    y_col_name : String
        Column name of Y values
    val_size_perc : Float
        Part of the df to use for Validation. 
        Format: [0.0;1.0]
    n_predictions : int, optional
        Number of future values to predict for Y, by default 5

    Returns
    -------
    smodel: json
        Fitted SARIMAX model on Y
    results: DataFrame
        DataFrame including the train, validation and forecast values from the SARIMAX fitted model on Y Time Series
    """

    X = df[y_col_name].values
    Y = df["Date"].values
    train_size = int(len(X) * (1 - val_size_perc))
    train, test = X[:train_size], X[train_size:len(X)]
    week = Y[train_size:len(X)]
    exo_past, exo_future = None, None

    # Split Exo Regressor into past (train + val) and future (forecast) values
    if exo is not None:
        exo_past, exo_future = exo[:len(X)], exo[len(X):len(exo)]

    # Create SARIMAX model or use input model
    print("Checking model for fit...")
    if model is None:
        print("No input model, starting to fit SARIMAX" + str(p) + str(d) +
              str(q) + str(P) + str(D) + str(Q) + str(s))
        smodel = pmdarima.arima.ARIMA(order=[p, d, q],
                                      method="lbfgs",
                                      maxiter=50,
                                      suppress_warnings=True)
        smodel = smodel.fit(df[y_col_name].values, exo_past)
        print("Finished SARIMAX fit.")
    else:
        print("Existing input model, will use it")
        smodel = model

    # Test model on the Validation set
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = sarimax.SARIMAX(history,
                                order=smodel.order,
                                seasonal_order=smodel.seasonal_order,
                                enforce_stationarity=False)
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

    # Add Train set to output
    data = pd.DataFrame()
    data["Date"] = Y[0:train_size]
    data["Predicted Net Order Value"] = None
    data["Actual Net Order Value"] = X[0:train_size]
    data["Classification"] = "train"

    # Add Validation set to output
    Tested = pd.DataFrame()
    Tested["Date"] = week
    Tested["Predicted Net Order Value"] = predictions
    Tested["Actual Net Order Value"] = test
    Tested["Classification"] = "test"
    Tested["Predicted Net Order Value"] = Tested[
        "Predicted Net Order Value"].astype(float)
    Tested["Date"] = pd.to_datetime(Tested["Date"])

    # Add Forecast set to output
    print("Predicting forecast values...")
    n_periods = n_predictions
    fitted, confint = smodel.predict(n_periods=n_periods,
                                     return_conf_int=True,
                                     exogenous=exo_future)
    print("Finished predicting forecast values.")
    rng = pd.date_range(df["Date"].max(), periods=n_periods, freq="7D")
    forecast = pd.DataFrame({
        "Date": rng,
        "Predicted Net Order Value": fitted,
        "Actual Net Order Value": None,
        "Classification": "forecast",
        "Conf_lower": confint[:, 0],
        "Conf_Upper": confint[:, 1],
    })
    forecast = forecast.drop(forecast.index[0])

    # Combine all sets
    results = data.append(Tested, ignore_index=True)
    results = results.append(forecast, ignore_index=True)
    results["Date"] = pd.to_datetime(results["Date"])
    # Reformat Dates to Date type
    results["Date"] = pd.to_datetime(results["Date"])
    return smodel, results


def predict_trend(train_df, reg, exo_col_name, pred_trend_col_name):
    """Trend Regression to predict future Trend"""
    # Get Regressor on prediction timeframe
    X_F = train_df[exo_col_name].dropna().values.reshape(-1, 1)
    print(X_F.shape)
    print(reg.predict(X_F).shape)
    # Predict Trend using fitted Regression on Regressor
    t_pred = reg.predict(X_F)
    len_pred = t_pred.shape[0]
    train_df["Predicted Trend"] = np.nan
    train_df["Predicted Trend"][-len_pred:] = t_pred.ravel()
    return X_F, train_df


def create_r_df(train_df, columns_to_drop, col_to_rename):
    """Create Residuals DataFrame"""
    r_df = train_df.copy()
    r_df = r_df.drop(columns=columns_to_drop)
    r_df = r_df.reset_index()
    r_df = r_df.rename(columns=col_to_rename)
    return r_df


def add_r(train_df, results_df_r, r_col_name, class_col_name):
    """Add Residuals (Train, Val and Forecast) to the Input Df"""
    results_df_r_idx = results_df_r.set_index("Date")
    train_df[r_col_name] = np.nan
    train_df[r_col_name] = results_df_r_idx["Predicted Net Order Value"]
    train_df[class_col_name] = results_df_r_idx[class_col_name]
    return train_df


def calc_y_pred(train_df, y_pred_col_name, trend_pred_col_name,
                seasonality_col_name, class_col_name):
    """Calculate Predicted Y with Predicted T, S and Predicted R components, on Validation and Forecast sets"""
    train_df[y_pred_col_name] = np.nan
    # Validation Y values
    mask = train_df[class_col_name] == "test"
    train_df.loc[mask, y_pred_col_name] = (train_df[trend_pred_col_name] +
                                           train_df[seasonality_col_name] +
                                           train_df["Predicted R"])
    # Future Y values
    mask = train_df[class_col_name] == "forecast"
    train_df.loc[mask, y_pred_col_name] = (train_df[trend_pred_col_name] +
                                           train_df[seasonality_col_name] +
                                           train_df["Predicted R"])
    return train_df
