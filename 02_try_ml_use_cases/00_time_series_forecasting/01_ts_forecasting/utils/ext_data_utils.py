import pandas as pd
from . import plot_utils


def load_external_data(
    overall_gdp_csv,
    industry_gdp_perc_csv,
    csv_col_1,
    csv_col_2,
    csv_col_2_new,
    col_final,
):
    """Load External/GDP data"""

    # Load National GDP data (need to create/upload external csv)
    all_gdp_csv = pd.read_csv(overall_gdp_csv)

    # Load Industry GDP % csv (need to create/upload external csv)
    all_gdp_ind_perc_csv = pd.read_csv(industry_gdp_perc_csv)
    # Rename col
    all_gdp_ind_perc_csv = all_gdp_ind_perc_csv.rename(
        columns={csv_col_2: csv_col_2_new})

    # Manually estimate GDP values for future quarters (CORE for TS Predictions)
    all_gdp = all_gdp_csv.copy()
    all_gdp = all_gdp.append([
        {
            "DATE": "7/1/2020",
            csv_col_1: 20200.0
        },
        {
            "DATE": "10/1/2020",
            csv_col_1: 21000.0
        },
        {
            "DATE": "1/1/2021",
            csv_col_1: 21000.0
        },
    ])
    all_gdp = all_gdp.reset_index(drop=True)

    # Manually estimate Industry GDP % values for future quarters (CORE for TS Predictions)
    all_gdp_ind_perc = all_gdp_ind_perc_csv.append([
        {
            "DATE": "4/1/2020",
            csv_col_2_new: 11.0
        },
        {
            "DATE": "7/1/2020",
            csv_col_2_new: 11.0
        },
        {
            "DATE": "10/1/2020",
            csv_col_2_new: 11.0
        },
        {
            "DATE": "1/1/2021",
            csv_col_2_new: 11.0
        },
    ])
    # Convert to %
    all_gdp_ind_perc[csv_col_2_new] = all_gdp_ind_perc[csv_col_2_new] / 100.0
    all_gdp_ind_perc = all_gdp_ind_perc.reset_index(drop=True)
    all_gdp_ind_perc.head()

    # Calculate Industry GDP
    all_gdp[col_final] = all_gdp[csv_col_1] * all_gdp_ind_perc[csv_col_2_new]

    # Resample to weekly GDP data
    all_gdp["DATE"] = pd.to_datetime(all_gdp["DATE"], format="%m/%d/%Y")
    all_gdp_weekly = all_gdp.copy()
    all_gdp_weekly = all_gdp_weekly.drop(columns=csv_col_1)
    all_gdp_weekly = all_gdp_weekly.set_index("DATE").resample(
        "W").ffill().reset_index()
    all_gdp_weekly[col_final] = all_gdp_weekly[col_final] * 4 / 52
    # Plot resampled external data
    plot_utils.plot_gdp(all_gdp_weekly, col_final)

    # Smoothen the weekly GDP data
    ext_data = all_gdp_weekly.copy()
    ext_data[col_final] = ext_data.iloc[:, 1].rolling(window=12,
                                                      center=False,
                                                      min_periods=1).mean()
    # Plot final external data
    plot_utils.plot_gdp(ext_data, col_final)
    return ext_data
