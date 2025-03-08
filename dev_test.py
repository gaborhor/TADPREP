import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tadprep as tp  # Testing package-level import - I want TADPREP to mirror Pandas in its common practice

from tadprep.core.transforms import PlotHandler

# import sys
# import os
# print(sys.path)
# print(os.environ.get('PYTHONPATH'))

# FIRST-PASS TESTING
# Uploading sample datafile
df_raw = pd.read_csv(r"C:\Users\gabor\Desktop\code\TADPREP\data\sample_data_sparse.csv")

# Checking import
# print(df_raw)


# Testing file_info method
# tp.file_info(df_raw, verbose=False)
# tp.file_info(df_raw, verbose=True)


# Testing reshape method
# df_reshape = tp.reshape(df_raw, verbose=False)
# df_reshape = tp.reshape(df_raw, verbose=True)
# print(df_reshape)

## Testing PlotHandler class
plot_handler = PlotHandler()
age_tuple = plot_handler.det_plot_type(df_raw, 'age')
race_tuple = plot_handler.det_plot_type(df_raw, 'race')
print(age_tuple)
print(race_tuple)

plot_handler.plot_data(df_raw, 'age')
plot_handler.plot_data(df_raw, 'race')
plot_handler.recall_plot('age', age_tuple[1])
plot_handler.recall_plot('race', race_tuple[1])
plot_handler.compare_plots('age')
plot_handler.compare_plots('race')

plot_handler.plot_data(df_raw, 'age')
plot_handler.plot_data(df_raw, 'race')
plot_handler.recall_plot('age', age_tuple[1])
plot_handler.recall_plot('race', race_tuple[1])
plot_handler.compare_plots('age')
plot_handler.compare_plots('race')


# Testing rename_and_tag method
# df_renamed = tp.rename_and_tag(df_raw, verbose=False, tag_features=False)
# df_renamed = tp.rename_and_tag(df_raw, verbose=True, tag_features=False)
# df_renamed = tp.rename_and_tag(df_raw, verbose=False, tag_features=True)
# df_renamed = tp.rename_and_tag(df_raw, verbose=True, tag_features=True)
# print(df_renamed)


# Testing feature_stats method
# tp.feature_stats(df_raw, verbose=False, summary_stats=False)
# tp.feature_stats(df_raw, verbose=True, summary_stats=False)
# tp.feature_stats(df_raw, verbose=False, summary_stats=True)
# tp.feature_stats(df_raw, verbose=True, summary_stats=True)


# Testing impute method
# df_impute = tp.impute(df_raw, verbose=False, skip_warnings=False)
# df_impute = tp.impute(df_raw, verbose=True, skip_warnings=False)
# df_impute = tp.impute(df_raw, verbose=False, skip_warnings=True)
# df_impute = tp.impute(df_raw, verbose=True, skip_warnings=True)
# print(df_impute)


# Testing encode method
# Note that this method must be tested with and without a passed list of features to encode
# encode_feats = ['sex', 'race', 'us_citizen']

# df_encode = tp.encode(df_raw, encode_feats, verbose=True, skip_warnings=True)
# df_encode = tp.encode(df_raw, encode_feats, verbose=True, skip_warnings=False)
# df_encode = tp.encode(df_raw, encode_feats, verbose=False, skip_warnings=True)
# df_encode = tp.encode(df_raw, encode_feats, verbose=False, skip_warnings=False)

# df_encode = tp.encode(df_raw, verbose=True, skip_warnings=True)
# df_encode = tp.encode(df_raw, verbose=True, skip_warnings=False)
# df_encode = tp.encode(df_raw, verbose=False, skip_warnings=True)
# df_encode = tp.encode(df_raw, verbose=False, skip_warnings=False)
# print(df_encode)


# Testing scale method
# Note that this method must be tested with and without a passed list of features to scale
# scale_feats = ['age']

# df_scaled = tp.scale(df_raw, scale_feats, verbose=True, skip_warnings=True)
# df_scaled = tp.scale(df_raw, scale_feats, verbose=True, skip_warnings=False)
# df_scaled = tp.scale(df_raw, scale_feats, verbose=False, skip_warnings=True)
# df_scaled = tp.scale(df_raw, scale_feats, verbose=False, skip_warnings=False)

# df_scaled = tp.scale(df_raw, verbose=True, skip_warnings=True)
# df_scaled = tp.scale(df_raw, verbose=True, skip_warnings=False)
# df_scaled = tp.scale(df_raw, verbose=False, skip_warnings=True)
# df_scaled = tp.scale(df_raw, verbose=False, skip_warnings=False)
# print(df_scaled)


# Final test step is checking the full prep_df method
# Start with creating a sample dataframe of smaller scale
# df_test = pd.DataFrame({
#     'name': ['Jack', 'Joe', 'Jill', 'Jane', 'John'],
#     'age': [29, 31, 37, 34, 41],
#     'sex': ['M', 'M', 'F', 'F', np.nan],
#     'married': [0, 0, 1, 1, 1],
#     'happy': [3, 4, 5, 2, 2],
#     'region': ['Northeast', 'Central', 'Central', np.nan, np.nan]
# })

# Check dataframe content
# print(df_test)

# Define feature lists to pass at method call
# prep_encode = ['sex', 'married']
# prep_scale = ['age']

# Test without any passed feature lists
# processed_df = tp.prep_df(df_test)

# Test with one passed feature list
# processed_df = tp.prep_df(df_test, prep_encode)

# Test with two passed feature lists
# processed_df = tp.prep_df(df_test, prep_encode, prep_scale)

# Check prepared dataframe
# print(processed_df)


# SECOND-PASS TESTING (checking method augmentations with specific corner cases)

'''
Testing augmented df_info method
Need to check for detection of:
- Duplicate instances
- Low-variance features
- Infinite values
- Non-null empty strings
'''
info_df = pd.DataFrame({
    'name': ['John', 'Joe', 'Jack', 'James', 'Jake', 'John'],  # Creating one duplicate row for 'John'
    'age': [25, 57, 17, 31, np.nan, 25],  # Make sure we have a missing value
    'weight': [155, 155, 155, 155, 155, 155],  # Should trip the low-variance check
    'height': [67, 70, np.inf, 72, 65, 67],  # Should trip the infinite values check
    'hair_color': ['brown', 'black', '', 'blonde', 'bald', 'brown']  # Should trip the non-null empty string check
})

# This should report on the duplicated instance, but should not display the follow-up warning
# tp.df_info(info_df, verbose=False)

# This should display all warnings
# tp.df_info(info_df, verbose=True)


'''
Testing new subset method
Need to test for:
- True random sampling in subset
- Seeded random sampling in subset
- Stratified random sampling in subset
- Time-boundary creation in subsetting timeseries data
- Whether methodology explanations work as intended
- Whether automatic time interval identification works properly
- Summary of subset results
- Differentiation of UX in verbose vs. non-verbose modes 
'''
# Build non-timeseries dataset with values which should make subsetting validation easy
# subset_normal = pd.DataFrame(
#     {'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#      'name': ['Adam', 'Alice', 'Alex', 'Adrian', 'Ashton', 'Amanda', 'Alice', 'Ashley', 'Aaron', 'Austin'],
#      'sex': ['M', 'F', 'F', 'M', 'M', 'F', 'F', 'F', 'M', 'M'],
#      'age': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]}
# )

# df_subset_random = tp.subset(subset_normal, verbose=True)  # Verbose true random subset
# df_subset_random = tp.subset(subset_normal, verbose=False)  # Minimal-output true random subset
# print(df_subset_random)  # Print check

# df_subset_seeded = tp.subset(subset_normal, verbose=True)  # Verbose seeded random subset
# print(df_subset_seeded)

# df_subset_strat = tp.subset(subset_normal, verbose=True)  # Verbose stratified subset
# df_subset_strat = tp.subset(subset_normal, verbose=False)  # Minimal-output stratified subset
# print(df_subset_strat)

# Testing at day level
# subset_ts = pd.DataFrame(
#     {'date': [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d %H:%M') for x in range(10)],
#      'river_volume': [101513, 175474, 167819, 130656, 110106, 113368, 109843, 119749, 110872, 196164]}
# )
# # df_subset_ts = tp.subset(subset_ts, verbose=True)  # Verbose time-bound subset
# df_subset_ts = tp.subset(subset_ts, verbose=False)  # Minimal-output time-bound subset
# print(subset_ts)
# print()
# print(df_subset_ts)

# Testing at hour level
# subset_ts = pd.DataFrame(
#     {'date': [(datetime.now() - timedelta(hours=x)).strftime('%Y-%m-%d %H:%M') for x in range(10)],
#      'river_volume': [101513, 175474, 167819, 130656, 110106, 113368, 109843, 119749, 110872, 196164]}
# )
# df_subset_ts = tp.subset(subset_ts, verbose=True)  # Verbose time-bound subset
# print(subset_ts)
# print()
# print(df_subset_ts)


'''
Testing augmented rename_and_tag method
Need to test for:
- Detection of invalid python identifiers (e.g. new feature name that starts with a number)
- Detection of problematic characters (e.g. new feature name that contains a special character)
- Detection of bad anti-pattern (e.g. new feature which is all uppercase or very short)
- Detection of python keyword duplication (e.g. new feature name is the same as a python keyword)
- Whether the method catches duplicate feature names
- Whether the method catches "All selected features already tagged as ordinal"
- Whether the method catches "All selected features already tagged as targets"
- Format and appearance of change summary table
- Whether the verbose and non-verbose UXs are functioning appropriately 
'''
# df_orig = pd.DataFrame({
#     'animal': ['Elephant', 'Rhino', 'Hippo', 'Cheetah', 'Lion'],
#     'size_ord': [3, 2, 2, 1, 1],
#     'eats_people_target': ['No', 'No', 'No', 'Yes', 'Yes']
# })
# print(df_orig)  # Print check
# df_rename = tp.rename_and_tag(df_orig, verbose=True, tag_features=False)  # Test verbose with no tagging
# df_rename = tp.rename_and_tag(df_orig, verbose=False, tag_features=False)  # Test minimal output with no tagging
# df_rename = tp.rename_and_tag(df_orig, verbose=True, tag_features=True)  # Test verbose with tagging
# df_rename = tp.rename_and_tag(df_orig, verbose=False, tag_features=True)  # Test minimal output with tagging
# print(df_rename)


'''
Testing augmented impute method
Need to test for:
- Detection of false-numerical features
- Detection of low-variance features
- Detection of high outlier counts
- Detection of highly-correlated feature pairs
- Detection of timeseries data
- Detection of high missingness rate
- Constant-value imputation
- Random-sample imputation
- Visualizations (both single-feature and before-after)
- Final imputation summary table
'''
# non_ts_data = {
#     # Numerical features with different characteristics
#     'normal': [100, 102, 98, 103, 97, 101, np.nan, 100, 102, 98, 100, np.nan, 101, 97, 103],
#     'skewed': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 2.0, 5.0, 15.0, 30.0, 50.0, 100.0],  # Extreme skew
#     'low_var': [50.01, 50.02, 50.01, 50.02, 50.02, 50.01, 50.02, 50.01, np.nan, 50.02, 50.01, 50.02, 50.01, 50.02,
#                 50.01],
#     'outliers': [105, 102, 98, 350, 101, 103, 99, 345, 101, np.nan, 100, 355, 102, 98, 101],
#     'corr_1': [2, 4, 6, 8, 10, 12, 14, 16, np.nan, 20, 22, 24, 26, 28, 30],
#     'corr_2': [2.1, 4.2, 6.1, 8.1, 10.2, 12.1, 14.2, 16.1, np.nan, 20.2, 22.1, 24.2, 26.1, 28.2, 30.1],
#
#     # Categorical feature
#     'cat_normal': ['A', 'B', np.nan, 'B', 'B', 'A', 'B', 'A', np.nan, 'B', 'A', 'B', np.nan, 'B', np.nan, ],
#
#     # False numerical features (actually categorical or ordinal)
#     'false_num_bin': [0, np.nan, 0, 1, 1, 0, 1, 0, 1, np.nan, 0, 1, 0, 1, 0],
#     'false_num_ord': [np.nan, 2, 3, 2, np.nan, 1, 4, 2, 3, 1, np.nan, 2, 4, 3, np.nan]
# }
#
# # Timeseries test data
# ts_data = {
#     'date': [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(15)],
#     'numeric_value': [100, np.nan, 98, 102, np.nan, 99, 101, 98, 102, np.nan, 105, 99, 101, np.nan, 103],
#     'categorical': ['A', 'B', np.nan, 'A', 'B', 'A', np.nan, 'B', 'A', 'B', np.nan, 'A', 'B', 'A', 'B']
# }

# Create test DataFrames
# df_normal = pd.DataFrame(non_ts_data)
# df_ts = pd.DataFrame(ts_data).sort_values('date').reset_index(drop=True)

# Test impute method on non-timeseries data
# df_normal_imputed = tp.impute(df_normal, verbose=True, skip_warnings=False)  # Full walkthrough
# df_normal_imputed = tp.impute(df_normal, verbose=False, skip_warnings=True)  # Super-streamlined
# print(df_normal_imputed)

# df_ts_imputed = tp.impute(df_ts, verbose=True, skip_warnings=False)  # Full walkthrough
# df_ts_imputed = tp.impute(df_ts, verbose=False, skip_warnings=True)  # Super-streamlined
# print(df_ts_imputed)


