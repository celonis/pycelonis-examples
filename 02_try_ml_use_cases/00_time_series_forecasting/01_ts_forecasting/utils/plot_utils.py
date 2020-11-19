import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots


def plot_clean_y(df, train_df, y_max):
    """Plot Pre-processed Y"""
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(df["Date"], df["Net Order Value"], c="c", label="Y Original")
    plt.plot(train_df["Date"], train_df["Net Order Value"], c="b", label="Y")
    plt.legend(loc="upper right")
    plt.axis([min(train_df["Date"]), max(train_df["Date"]), 0, y_max])
    plt.show()


def plot_gdp(ext_data, col_final):
    """Plot resulting Industry GDP"""
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(ext_data["DATE"], ext_data[col_final], c="b")
    plt.show()


def plot_y_trend(train_df, t, y_min, y_max):
    """Plot Y and Trend"""
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(train_df["Date"], t, color="b", label="Trend")
    plt.plot(train_df["Date"],
             train_df["Net Order Value"],
             color="g",
             label="Y")
    plt.legend(loc="upper right")
    ax.set_ylim([y_min, y_max])
    plt.show()


def plot_y_trend_ext(train_df, Y, exo_col_name, exo_pretty_name, y_min, y_max,
                     y_min_exo, y_max_exo):
    """Plot Y, Trend and Exo Regressors"""
    fig, ax = plt.subplots(figsize=(20, 10))
    ax2 = ax.twinx()
    # Net Order Value
    ax.plot(train_df.index, train_df["Net Order Value"], color="g", label="Y")
    # External data/GDP
    ax2.plot(train_df.index,
             train_df[exo_col_name],
             color="c",
             label=exo_pretty_name)
    # Trend
    ax.plot(train_df.dropna().index[:len(Y)], Y, color="b", label="Trend")
    plt.legend(loc="upper right")
    ax.set_ylim([y_min, y_max])
    ax2.set_ylim([y_min_exo, y_max_exo])
    plt.show()


def plot_y_pred_trend_ext(train_df, exo_col_name, X, Y, X_F, y_min, y_max,
                          y_min_exo, y_max_exo):
    """Plot Predicted Y, Trend and Exo Regressors"""
    fig, ax = plt.subplots(figsize=(20, 10))
    ax2 = ax.twinx()
    # External Data/GDP
    ax2.plot(train_df[exo_col_name].dropna().index,
             train_df[exo_col_name].dropna(),
             color="m",
             label="External data (Full)")
    ax2.plot(train_df.dropna().index[:len(X)],
             X,
             color="c",
             label="External data (Train for Trend fit)")
    # Trend
    ax.plot(train_df.dropna().index[:len(Y)],
            Y,
            color="b",
            label="Trend (Train for Trend fit)")
    # Predicted Trend (through Reg)
    ax.plot(train_df[exo_col_name].dropna().index,
            train_df["Predicted Trend"][-len(X_F):],
            color="g",
            label="Trend (Predicted)")
    ax.legend(loc="upper right")
    ax2.legend(loc="lower right")
    ax.set_ylim([y_min, y_max])
    ax2.set_ylim([y_min_exo, y_max_exo])
    plt.show()


def plot_y_t_s(train_df, trend_col_name, seasonality_col_name):
    """Plot Y, T and S"""
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(train_df.index, train_df["Net Order Value"], color="g", label="Y")
    plt.plot(train_df.index, train_df[trend_col_name], color="b", label="T")
    plt.plot(train_df.index,
             train_df[trend_col_name] + train_df[seasonality_col_name],
             color="m",
             label="T+S")
    plt.legend(loc="upper right")
    plt.show()


def plot_y_t_s_with_pred(train_df, trend_col_name, seasonality_col_name,
                         pred_trend_col_name):
    """Plot Y, T, S and Predicted T + S"""
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(train_df.index, train_df["Net Order Value"], color="g", label="Y")
    plt.plot(train_df.index, train_df[trend_col_name], color="b", label="T")
    plt.plot(train_df.index,
             train_df[pred_trend_col_name],
             color="c",
             label="T Pred")

    plt.plot(train_df.index,
             train_df[trend_col_name] + train_df[seasonality_col_name],
             color="m",
             label="T+S")
    plt.plot(train_df.index,
             train_df[pred_trend_col_name] + train_df[seasonality_col_name],
             color="r",
             label="T Pred + S")
    plt.legend(loc="upper right")
    plt.show()


def plot_r(train_df, r_col_name):
    """Plot Residuals"""
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(train_df.index, train_df[r_col_name], color="y", label="R")
    plt.legend(loc="upper right")
    plt.show()


def plot_acf_pacf_r(r, lags):
    """Plot ACF and PACF plots for Residuals"""
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    fig = tsaplots.plot_acf(r.dropna(), lags=lags, ax=ax[0])
    fig = tsaplots.plot_pacf(r.dropna(), lags=lags, ax=ax[1])
    plt.show()


def plot_results(results):
    """Plot all Residual sets (train, Val and Forecast)"""
    fig, ax = plt.subplots(figsize=(20, 10))
    results["Date"] = results["Date"].astype(str)
    plt.plot(results["Date"], results["Predicted Net Order Value"], c="b")
    plt.plot(results["Date"], results["Actual Net Order Value"], c="r")
    plt.fill_between(results["Date"],
                     results["Conf_lower"],
                     results["Conf_Upper"],
                     color="k",
                     alpha=0.15)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_rotation(45)
        tick.set_visible(False)
        if i % 3 == 0:
            tick.set_visible(True)
    plt.show()


def plot_final(train_df, trend_col_name, seasonality_col_name, r_col_name,
               trend_pred_col_name, y_pred_col_name, class_col_name):
    """Plot Y, T, S, R and Predicted Y with intermediary components"""
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.plot(train_df.index, train_df["Net Order Value"], color="g", label="Y")
    plt.plot(train_df.index, train_df[trend_col_name], color="b", label="T")
    plt.plot(train_df.index,
             train_df[trend_col_name] + train_df[seasonality_col_name],
             color="m",
             label="T+S")
    # Seasonality
    plt.plot(train_df.index,
             train_df[seasonality_col_name],
             color="m",
             label="S")
    # Predicted Trend
    plt.plot(train_df.index,
             train_df[trend_pred_col_name],
             color="y",
             label="T Pred")
    plt.plot(train_df.index,
             train_df[trend_pred_col_name] + train_df[seasonality_col_name],
             color="k",
             label="T Pred + S")
    # Predicted Y on Validation part
    plt.plot(
        train_df[train_df[class_col_name] == "test"].index,
        train_df[train_df[class_col_name] == "test"][y_pred_col_name],
        color="c",
        label="Y Pred (val)",
    )
    # Predicted Y on Future part
    plt.plot(
        train_df[train_df[class_col_name] == "forecast"].index,
        train_df[train_df[class_col_name] == "forecast"][y_pred_col_name],
        color="r",
        label="Y Pred (future)",
    )
    plt.legend(loc="upper right")
    plt.show()
