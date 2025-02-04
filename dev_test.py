import pandas as pd
import tadprep as tp  # Testing package-level import - I want TADPREP to mirror Pandas in its common practice

# Uploading sample datafile
df_raw = pd.read_csv(r"C:\Users\doncs\Documents\GitHub\TADPREP\data\sample_data_longlean.csv")

# Checking import
# print(df_raw)

# Testing file_info method
# tp.file_info(df_raw, verbose=False)
# tp.file_info(df_raw, verbose=True)

# After debugging, I am satisfied with the operation of the file_info method

# Testing reshape method
# df_reshape = tp.reshape(df_raw, verbose=False)
# df_reshape = tp.reshape(df_raw, verbose=True)
# print(df_reshape)

# After debugging, I am satisfied with the operation of the reshape method
