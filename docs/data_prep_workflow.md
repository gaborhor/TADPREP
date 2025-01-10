# TADPREPS Data Preparation Workflow

**Note:** The flow of the script will be procedural from top to bottom and organized at the function level.

### Logging
- Set up logging file as top-level code at the beginning of the script

### Function 1: File Ingestion
- Specify to user that the script only processes .csv and Excel files 
- Fetch absolute filepath from user
- Error handling: Use pathlib library to check whether the file exists
- Error handling: Use pathlib library to check whether the file is larger than 1GB
- Use pathlib library to check whether the file is .csv or Excel
- Error handling: Raise error if the file exists but is of an unsupported type
- Load file into Pandas dataframe

### Function 2: Print Top-Level Tabular File Information
- Print number of rows (instances) in file
- Print number of columns (features) in file
- Print names and datatypes of columns/features in file
- Print number of instances/rows with any missing values

### Function 3: Allow User to 'Trim' the File
- Ask if the user wants to delete all instances/rows with *any* missing values
- Ask if the user wants to delete any columns/features
- Ask if the user wants to reduce the number of instances/rows by deleting a user-supplied percentage of rows at random (i.e. sub-setting the data)

### Function 4: Allow User to Rename Columns/Features
- Print the names of each field and allow the user to rename specified features
- Ask if the user is aware of their target field(s) and offer to append '_target' to the names of those features
- Ask if the user is aware of any ordinal-level fields and offer to append '_ord' to the names of those features

### Function 5: Display Top-Level Descriptive Statistics at the Feature-Type Level (Before Encoding or Scaling)
- For the categorical features:
  - Generate and store a list of the categorical features (Do not include any target features)
  - Display feature names (i.e. "The categorical features are: ...")
  - Display count and rate of missing values for each categorical feature
  - Display value counts, mean, median, and mode for each categorical feature
- For the numerical features:
  - Generate and store a list of the numerical features (Do not include any target features)
  - Display feature name and type (i.e. "The numerical features are: ...")
  - Display count and rate of missing values for each numerical feature
  - Display mean, median, and mode for each numerical feature

### Function 6: Impute Missing Values (If Desired)
- For each feature:
  - Print count and rate of instances/rows with missing values for that feature
  - If missing rate is over 10%, warn against imputation because of bias introduction
  - If missing rate is under 1%, warn against imputation because 'why bother'
  - Print options for simple imputation (mean, median, mode)
  - Warn user that more sophisticated imputation (i.e. imputation by modeling) is outside the scope of TADPREPS
  - Ask which imputation method (if any) the user wants to apply to the feature
  - Apply selected imputation method (or none, if user doesn't want to impute) to the feature

### Function 7: Facilitate Encoding of Categorical Features
- For each categorical feature:
  - Print total # of categories/values in the feature
  - Print alphabetized unique list of categories/values in the feature
  - Print warning message that if the number of categories in a categorical feature is too high, the user should perform some kind of dimensionality reduction rather than encode it
  - Print value counts for the feature
  - Ask if the user wants to see a barplot of the feature distribution
  - Ask if the user wants to one-hot encode or dummy the feature - provide very simple reminder of the difference
  - Apply selected encoding method (or none, if user doesn't want to encode) to the feature
  - Do the same (with ordinal encoding) for any ordinal features
  - Return the encoded dataset

### Function 8: Display Plots for Numerical Features and Facilitate Scaling
- For each numerical feature:
  - Print simple descriptive statistics of the feature
  - Ask if the user wants to see a histogram of the feature distribution
  - Print a very simple reminder of the different common scaling methods and when each should be used
  - Ask which scaling method (if any) the user wants to apply to the feature
  - Apply selected scaling method (or none, if user doesn't want to scale) to the feature

### Function 9: Export Prepared Dataset
- Ask user for preferred export format (list of eligible/suggested formats apart from Excel and .csv is TBD)
- Ask user what they want the file to be called
- Ask user where they want the file to be saved (make directory if it doesn't already exist)
- Export file in desired format
