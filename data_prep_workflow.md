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
- Print number of missing values at the feature level - both total count and '% missing'
- Print warning message re: imputation if a feature's '% missing' value is over 10%

### Function 3: Allow User to 'Trim' the File
- Ask if the user wants to delete all rows with *any* missing values
- Ask if the user wants to delete any columns/features
- Ask if the user wants to reduce the number of rows by deleting a user-supplied percentage of rows at random (i.e. Sub-setting the data)

### Function 4: Allow User to Rename Columns/Features
- Print the names of each field and allow the user to rename specified features
- Ask if the user is aware of their target field(s) and offer to append '_target' to the names of those features
- Ask if the user is aware of any ordinal-level fields and offer to append '_ord' to the names of those features

### Function 5: Display Top-Level Descriptive Statistics and Plots at the Feature Level (Before Encoding or Scaling)
- TBC

### Function 6: Facilitate One-Hot Encoding of Categorical Features
- TBC

### Function 7: Display Plots for Numerical Features and Facilitate Scaling
- TBC

### Function 8: Export Prepared Dataset
- TBC