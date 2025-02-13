import numpy as np
import pandas as pd
import tadprep as tp  # Testing package-level import - I want TADPREP to mirror Pandas in its common practice

# FIRST-PASS TESTING
# Uploading sample datafile
df_raw = pd.read_csv(r"C:\Users\doncs\Documents\GitHub\TADPREP\data\sample_data_longlean.csv")


# Checking import
# print(df_raw)


# Testing file_info method
# tp.file_info(df_raw, verbose=False)
# tp.file_info(df_raw, verbose=True)


# Testing reshape method
# df_reshape = tp.reshape(df_raw, verbose=False)
# df_reshape = tp.reshape(df_raw, verbose=True)
# print(df_reshape)


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
