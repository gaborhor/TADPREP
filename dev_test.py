import pandas as pd
import tadprep as tp  # Testing package-level import - I want TADPREP to mirror Pandas in its common practice

# Uploading sample datafile
df_raw = pd.read_csv(r"C:\Users\doncs\Documents\GitHub\TADPREP\data\sample_data_longlean.csv")

# Checking import
print(df_raw)


