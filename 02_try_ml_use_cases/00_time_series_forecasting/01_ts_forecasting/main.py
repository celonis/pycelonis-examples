import pandas as pd
from pycelonis import get_celonis, pql

from utils import ext_data_utils, model_utils, utils


# Load input data
celonis = get_celonis()
datamodel = celonis.datamodels.find("dm_id")
input_columns = [("col_name", "pretty_name"), ("col_name_2", "pretty_name_2")]
input_filter = "FILTER TBD"

query = pql.PQL()
for col_name, col_pretty_name in input_columns:
    query += pql.PQLColumn(col_name, col_pretty_name)
query += pql.PQLFilter(input_filter)
df = datamodel.get_data_frame(query)

# Import External Data for n-step Predictions (such as GDP below)
ext_data = ext_data_utils.load_external_data(
    overall_gdp_csv="US_GDP.csv",
    industry_gdp_perc_csv="US_MANUF_GDP_PERC.csv",
    csv_col_1="GDP",
    csv_col_2="VAPGDPMA",
    csv_col_2_new="IND_PERC",
    col_final="IND_GDP",
)

# INPUT Product Families
# OUTPUT Exported Predictions for DM

# INPUTS
subsets = ["Subset1", "Subset2"]
subset_needs_adjusts = ["Subset2"]
subset_col_name = "TBD"
y_col_name = "Y Value"
val_size_perc = 0.2
# OUTPUTS
all_subset_results = {}
all_subset_exports = {}
output_col_names = {
    "index": "Date",
    y_col_name: "Actual Y Value",
    y_pred_col_name: "Predicted Y Value",
    r_class_col_name: "Classification",
}

# Run Predictions for each selected subset
for subset in subsets:
    print("Running model for ", subset)
    # Check if subset needs baseline adjustment
    to_adjust = False
    if subset in subset_needs_adjusts:
        to_adjust = True

    # Filter train df for subset
    subset_train_df = train_df[train_df[subset_col_name] == prod_fam]
    subset_train_df.drop(columns=[subset_col_name], inplace=True)
    # Run Predictions model for this subset
    subset_results = model_utils.run_predictions_model(fm_train_df, ext_data, to_adjust)
    # Store Output (subset Predictions)
    all_subset_results[subset] = subset_results
    print(subset, all_subset_results[subset].shape)
    # Store export-version of the Output (subset Predictions)
    all_subset_exports[subset] = utils.prepare_export_df(subset_results, output_col_names, y_pred_col_name)

print("Finished running predictions for all subsets, total output shape is ", all_subset_results[subset].shape)
print("Subsets are ", all_subset_exports.keys())

# Combine Results into single Export table
# Add new 'subset name' column to the export-version of Predictions
export_df = pd.DataFrame()
for key in all_subset_exports:
    print("Adding ", key, " value in new column")
    subset_df = all_subset_exports[key]
    subset_df[subset_col_name] = key
    print("shape of subset export-version is ", subset_df.shape)
    export_df = pd.concat([export_df, subset_df], axis=0)

# Export table to DM
datamodel_export = celonis.datamodels.find("dm_id")
dm_export_table_name = "Predictions_Output"
datamodel_export.push_table(export_df, dm_export_table_name, reload_datamodel=False, if_exists="replace")
