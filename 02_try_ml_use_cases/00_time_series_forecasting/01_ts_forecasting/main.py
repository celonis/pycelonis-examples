import pandas as pd
from pycelonis import get_celonis, pql

from utils import ext_data_utils, model_utils, utils

# Load input data
celonis = get_celonis()
dm_id = 'TBD'
datamodel = celonis.datamodels.find(dm_id)
input_columns = [("col_name", "pretty_name"), ("col_name_2", "pretty_name_2")]
input_filter = "FILTER TBD"

train_df = utils.get_pql_dataframe(datamodel, input_columns, input_filter)

# Import External Data for n-step Predictions (such as GDP below)
ext_data = ext_data_utils.load_external_data(
    overall_gdp_csv="US_GDP.csv",
    industry_gdp_perc_csv="US_MANUF_GDP_PERC.csv",
    csv_col_1="GDP",
    csv_col_2="VAPGDPMA",
    csv_col_2_new="IND_PERC",
    col_final="IND_GDP",
)

# INPUTS
subsets = ['subset1', 'subset2']  # PARAM
subset_needs_adjusts = ['subset2'
                       ]  # PARAM Subsets which need a baseline adjustment
subset_col_name = 'subset_filtering_column'  # PARAM
input_y_col_name = "Y_column"  # PARAM
input_exo_col_name = 'ext_data_column'  # PARAM
model_class_col_name = 'classification_naming'  # PARAM Column to flag train vs test vs forecast timeframes
model_y_pred_col_name = 'Y_prediction_column'  # PARAM
val_size_perc = 0.2

# OUTPUTS, for Exported Predictions to DM
all_subset_results = {}
all_subset_exports = {}
output_col_names = {
    "index": "Date",  # PARAM
    input_y_col_name: "Actual Y Value",  # PARAM
    model_y_pred_col_name: "Predicted Y Value",  # PARAM
    model_class_col_name: "Classification",  # PARAM
}

# Run Predictions for each selected subset
for subset in subsets:
    # Check if subset needs baseline adjustment
    to_adjust = False
    if subset in subset_needs_adjusts:
        to_adjust = True

    # Filter train df for subset
    subset_train_df = utils.get_subset_df(train_df, subset, subset_col_name)

    # Run Predictions model for this subset
    print('Run TS Predictions model for subset train df \n',
          subset_train_df.head())
    subset_results = model_utils.run_predictions_model(subset_train_df,
                                                       ext_data,
                                                       input_y_col_name,
                                                       input_exo_col_name,
                                                       val_size_perc, to_adjust)
    # Store Output (subset Predictions)
    all_subset_results[subset] = subset_results
    print('subset ', subset, ' Prediction outputs have shape ',
          all_subset_results[subset].shape)
    # Store export-version of the Output (subset Predictions)
    all_subset_exports[subset] = utils.prepare_export_df(
        subset_results, output_col_names, model_y_pred_col_name)

print("Finished running predictions for all subsets, total output shape is ",
      all_subset_results[subset].shape)
print("Subsets are ", all_subset_exports.keys())

# Combine Results into single Export table
# Add new 'subset name' column to the export-version of Predictions
export_df = utils.constitute_export_df(all_subset_exports, subset_col_name)

# Export table to DM
export_table_name = "Predictions_Output"
print('Export df shape is ', export_df.shape)
print('Export df head is ')
print(export_df.head(10))
print('Export df tail is ')
print(export_df.tail(10))
datamodel.push_table(export_df,
                     export_table_name,
                     reload_datamodel=False,
                     if_exists="replace")
