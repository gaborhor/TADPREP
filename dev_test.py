import pandas as pd
import tadprep as tp  # Testing package-level import - I want TADPREP to mirror Pandas in its common practice

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
