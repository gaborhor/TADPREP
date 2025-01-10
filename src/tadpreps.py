"""
This is the TADPREPS codebase. All core functions will be defined internally (so no additional files are necessary at
runtime) and the logging file will be created in the same working directory as the script itself.
"""

# Library imports
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import seaborn as sns
import logging
import sys

# Fetch current runtime timestamp in a readable format
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# Set up error logging with time-at-execution
# We persist with "%-type" formatting to preserve backward compatibility
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(f'tadpreps_runtime_log_{timestamp}.log'), logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
logger.info('Initiating TADPREPS...')


def load_file() -> pd.DataFrame:
    """
    This function gets the absolute filepath from the user, performs some verification checks on the file, and if the
    checks are passed, loads the tabular data into a Pandas dataframe.
    Args:
        None. This is a nullary function.
    Returns:
        df_full (pd.DataFrame): A Pandas dataframe containing the full, unaltered dataset.
    """
    print('NOTE: TADPREPS supports only .csv and Excel files.')

    try:
        # Fetch filepath and convert to Path object
        filepath = Path(input('Enter the absolute path to your datafile: ')).resolve()

        # Error handling: Use pathlib library to check whether the supplied path is valid
        if not filepath.exists():
            logger.error(f'The supplied path "{filepath}" is invalid.')
            logger.error('Please resolve this issue and re-run TADPREPS.')
            sys.exit(1)

        # Error handling: Use pathlib library to check whether the path leads to a directory rather than a file
        if not filepath.is_file():
            logger.error(f'The supplied path "{filepath}" does not point to a single file.')
            logger.error('Please resolve this issue and re-run TADPREPS.')
            sys.exit(1)

        # Error handling: Raise error if the file exists but is of an unsupported type
        if filepath.suffix.lower() not in ['.csv', '.xls', '.xlsx']:
            logger.error('TADPREPS only supports .csv, .xls, and .xlsx files.')
            logger.error(f'The file at "{filepath}" does not appear to be of a compatible type.')
            logger.error('Please resolve this issue and re-run TADPREPS.')
            sys.exit(1)

        # Error handling: Use pathlib library to check whether the file is larger than 1GB
        size_mb = filepath.stat().st_size / (1024 ** 2)  # Fetch file size, divide by 1024^2 to get size in megabytes
        if size_mb > 1000:
            logger.error(f'File size ({size_mb} megabytes) exceeds 1 GB limit.')
            logger.error('For files of this size, consider an out-of-memory or distributed solution.')
            sys.exit(1)

        # Use pathlib library to check whether the file is .csv or Excel and load into Pandas dataframe
        if filepath.suffix.lower() == '.csv':
            file_type = 'CSV'
            df_full = pd.read_csv(filepath)
        else:
            file_type = 'Excel'
            df_full = pd.read_excel(filepath)

        logger.info(f'Successfully loaded {file_type} file at {filepath}.')
        logger.info(f'Base shape of file: {df_full.shape}')
        return df_full

    except Exception as exc:
        logger.error(f'An unexpected error occurred: {exc}')
        sys.exit(1)


def print_file_info(df_full: pd.DataFrame) -> None:
    """
    This function prints and logs use information about the full, unaltered datafile for the user.
    Args:
        df_full (pd.DataFrame): A Pandas dataframe containing the full, unaltered dataset.
    Returns:
        None. This is a void function.
    """
    # Print number of rows (instances) in file
    logger.info(f'The unaltered file has {df_full.shape[0]} rows/instances.')  # [0] is rows

    # Print number of columns (features) in file
    logger.info(f'The unaltered file has {df_full.shape[1]} columns/features.')  # [1] is columns

    # Print names and datatypes of columns/features in file
    logger.info('Names and datatypes of columns/features:')
    logger.info(df_full.info(memory_usage=False, show_counts=False))  # Limit information printed since it's logged

    # Print # of instances/rows with missing values
    row_missing_cnt = df_full.isnull().any(axis=1).sum()  # Compute count
    row_missing_rate = (row_missing_cnt / len(df_full) * 100).round(2)  # Compute rate
    logger.info(f'\n{row_missing_cnt} rows/instances ({row_missing_rate}%) contain at least one missing value.')


def trim_file(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    This function allows the user to delete instances with any missing values, to drop columns/features from the
    dataset, and to sub-set the data by deleting a specified proportion of the instances at random.
    Args:
        df_full (pd.DataFrame): The original, unaltered dataset.
    Returns:
        df_trimmed (pd.DataFrame): The dataset after trimming/sub-setting.
    """
    df_trimmed = df_full.copy(deep=True)
    # Ask if the user wants to delete *all* instances with any missing values
    user_drop_na = input('Do you want to drop all rows/instances with *any* missing values? (Y/N): ')
    if user_drop_na.lower() == 'y':
        df_trimmed = df_trimmed.dropna()
        logger.info(f'After deletion of instances with missing values, {len(df_trimmed)} instances remain.')

    # Ask if the user wants to drop any of the columns/features in the dataset
    user_drop_cols = input('Do you want to drop any of the columns/features in the dataset? (Y/N): ')
    if user_drop_cols.lower() == 'y':
        print('The full set of columns/features in the dataset is:')
        for col_idx, column in enumerate(df_trimmed.columns, 1):  # Create enumerated list of features starting at 1
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                drop_cols_input = input('\nEnter the index integers of the columns/features you wish to drop '
                                     '(comma-separated) or enter "C" to cancel: ')

                # Check for user cancellation
                if drop_cols_input.lower() == 'c':  # If user entered cancel-out input
                    logger.info('Column/feature deletion cancelled.')  # Log the cancellation
                    break  # Exit the while loop

                # Create list of column indices to drop
                drop_cols_idx = [int(idx.strip()) for idx in drop_cols_input.split(',')]  # Splitting on comma

                # Verify that all index numbers of columns to be dropped are valid/in range
                if not all(1 <= idx <= len(df_trimmed.columns) for idx in drop_cols_idx):  # Using a generator
                    raise ValueError('Some column/feature index integers entered are out of range/invalid.')

                # Convert specified column numbers to actual column names
                drop_cols_names = [df_trimmed.columns[idx-1] for idx in drop_cols_idx]  # Subtracting 1 from indices

                # Drop the columns
                df_trimmed.drop(columns=drop_cols_names, inplace=True)
                logger.info(f'Dropped columns/features: {",".join(drop_cols_names)}')  # Log the dropped columns
                break  # Exit the while loop

            # Catch invalid user input
            except ValueError:
                logger.error('Invalid input. Please enter valid column/index integers separated by commas.')
                continue  # Restart the loop

    # Ask if the user wants to sub-set the data by deleting a specified proportion of the instances at random
    user_subset = input('Do you want to sub-set the data by randomly deleting a specified proportion of '
                        'rows/instances? (Y/N): ')
    if user_subset.lower() == 'y':
        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                subset_input = input('Enter the proportion of rows/instances to DROP (0.0-1.0) or '
                                     'enter "C" to cancel: ')

                # Check for user cancellation
                if subset_input.lower() == 'c':  # If user entered cancel-out input
                    logger.info('Random sub-setting cancelled.')  # Log the cancellation
                    break  # Exit the while loop

                subset_rate = float(subset_input)  # Convert string input to float
                if 0 < subset_rate < 1:  # If the float is valid (i.e. between 0 and 1)
                    retain_rate = 1 - subset_rate  # Compute retention rate
                    retain_row_cnt = int(len(df_trimmed) * retain_rate)  # Select count of rows to keep in subset

                    df_trimmed = df_trimmed.sample(n=retain_row_cnt)  # No random state set b/c we want true randomness
                    logger.info(f'Randomly dropped {subset_rate}% of rows/instances. {retain_row_cnt} rows/instances '
                                f'remain.')  # Log sub-setting information/outcome
                    break  # Exit while loop

                # Catch user input error for invalid/out-of-range float
                else:
                    logger.error('Enter a value between 0.0 and 1.0.')

            # Catch outer-level user input errors
            except ValueError:
                logger.error('Invalid input. Enter a float value between 0.0 and 1.0 or enter "C" to cancel.')
                continue  # Restart the loop

    return df_trimmed  # Return the trimmed dataframe


def rename_features(df_trimmed: pd.DataFrame) -> pd.DataFrame:
    """
    This function allows the user to rename columns/features and to append the '_ord' and/or '_target' suffixes
    to ordinal or target columns.
    Args:
        df_trimmed (pd.DataFrame): The 'trimmed' dataset created by trim_file().
    Returns:
        df_renamed (pd.DataFrame): The dataset with renamed columns.
    """
    df_renamed = df_trimmed.copy(deep=True)  # Create copy of trimmed dataset

    # Ask if user wants to rename any columns
    user_rename_cols = input('Do you want to rename any of the columns/features in the dataset? (Y/N): ')
    if user_rename_cols.lower() == 'y':
        print('The list of columns/features currently in the dataset is:')
        for col_idx, column in enumerate(df_renamed.columns, 1):  # Create enumerated list of features starting at 1
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                rename_cols_input = input('\nEnter the index integer of the column/feature you wish to rename '
                                     'or enter "C" to cancel: ')

                # Check for user cancellation
                if rename_cols_input.lower() == 'c':  # If user entered cancel-out input
                    logger.info('Column/feature renaming cancelled.')  # Log the cancellation
                    break  # Exit the while loop

                col_idx = int(rename_cols_input)  # Convert input to integer
                if not 1 <= col_idx <= len(df_renamed.columns):  # Validate entry
                    raise ValueError('Column index is out of range.')

                # Get new name for the column from user
                col_name_old = df_renamed.columns[col_idx - 1]
                col_name_new = input(f'Enter new name for column "{col_name_old}": ').strip()

                # Validate name to make sure it doesn't already exist in the dataset
                if col_name_new in df_renamed.columns:
                    logger.error(f'Column name "{col_name_new}" already exists. Choose a different name.')
                    continue  # Restart the loop

                # Rename column in-place
                df_renamed.rename(columns={col_name_old: col_name_new}, inplace=True)
                logger.info(f'Renamed column "{col_name_old}" to "{col_name_new}".')

                # Ask if user wants to rename another column
                if input('Do you want to rename another column? (Y/N): ').lower() != 'y':
                    break  # If not, exit the while loop

            except ValueError as exc:
                logger.error(f'Invalid input: {exc}')

    # Ask if user wants to perform batch-level appending of '_ord' tag to ordinal features
    user_tag_ord = input('Do you want to tag any columns/features as ordinal by appending the "_ord" suffix '
                              'to their names? (Y/N): ')

    if user_tag_ord.lower() == 'y':
        print('\nCurrent column/feature list:')
        for col_idx, column in enumerate(df_renamed.columns, 1):
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                ord_input = input('\nEnter the index integers of ordinal features (comma-separated) '
                                  'or enter "C" to cancel: ')

                if ord_input.lower() == 'c':  # If user cancels
                    logger.info('Ordinal feature tagging cancelled.')  # Log the cancellation
                    break  # And exit the while loop

                ord_idx_list = [int(idx.strip()) for idx in ord_input.split(',')]  # Create list of index integers

                # Validate that all entered index integers are in range
                if not all(1 <= idx <= len(df_renamed.columns) for idx in ord_idx_list):  # Using a generator again
                    raise ValueError('Some column/feature indices are out of range.')

                ord_names_pretag = [df_renamed.columns[idx - 1] for idx in ord_idx_list]  # Create list of pretag names

                # Generate mapper for renaming columns with '_ord' suffix
                ord_rename_map = {name: f'{name}_ord' for name in ord_names_pretag if not name.endswith('_ord')}

                # Validate that tags for the selected columns are not somehow already present (i.e. done pre-import)
                if not ord_rename_map:  # If the mapper is empty
                    logger.warning('All selected columns/features are already tagged as ordinal.')  # Warn the user
                    break  # And exit the while loop

                df_renamed.rename(columns=ord_rename_map, inplace=True)  # Perform tagging
                logger.info(f'Tagged the following columns/features as ordinal: {", ".join(ord_rename_map.keys())}')
                break

            # Catch invalid input
            except ValueError as exc:
                logger.error(f'Invalid input: {exc}')
                continue  # Restart the loop

    # Ask if user wants to perform batch-level appending of '_target' tag to target features
    user_tag_target = input('Do you want to tag any columns/features as targets by appending the "_target" suffix '
                            'to their names? (Y/N): ')

    if user_tag_target.lower() == 'y':
        print('\nCurrent columns:')
        for col_idx, column in enumerate(df_renamed.columns, 1):
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                target_input = input('\nEnter the index integers of target features (comma-separated) '
                                     'or enter "C" to cancel: ')

                if target_input.lower() == 'c':  # If user cancels
                    logger.info('Target feature tagging cancelled.')  # Log the cancellation
                    break  # And exit the while loop

                target_idx_list = [int(idx.strip()) for idx in target_input.split(',')]  # Create list of index integers

                # Validate that all entered index integers are in range
                if not all(1 <= idx <= len(df_renamed.columns) for idx in target_idx_list):  # Using a generator again
                    raise ValueError('Some column/feature indices are out of range.')

                target_names_pretag = [df_renamed.columns[idx - 1] for idx in target_idx_list]  # List of pretag names

                # Generate mapper for renaming columns with '_target' suffix
                target_rename_map = {name: f'{name}_target' for name in target_names_pretag if
                                     not name.endswith('_target')}

                # Validate that tags for the selected columns are not somehow already present (i.e. done pre-import)
                if not target_rename_map:  # If the mapper is empty
                    logger.warning('All selected columns/features are already tagged as targets.')  # Warn the user
                    break  # And exit the while loop

                df_renamed.rename(columns=target_rename_map, inplace=True)  # Perform tagging
                logger.info(f'Tagged the following columns/features as targets: {", ".join(target_rename_map.keys())}')
                break

            # Catch invalid input
            except ValueError as exc:
                logger.error(f'Invalid input: {exc}')
                continue  # Restart the loop

    return df_renamed  # Return dataframe with renamed and tagged columns


def print_feature_stats(df_renamed: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    This function aggregates the features by class (i.e. feature type) and prints top-level missingness and
    descriptive statistics information at the feature level. It then builds and logs summary tables at the feature-class
    level.
    Args:
        df_renamed (pd.DataFrame): The renamed/tagged dataframe created by rename_features().
    Returns:
        A tuple of lists of strings for the non-target features at the feature-class level. These are used by the
        scaling and encoding functions.
    """
    logger.info('Displaying top-level information for columns/features in dataset...')

    # Create a list of columns which are categorical and do NOT have the '_ord' or '_target' suffixes
    cat_cols = [column for column in df_renamed.columns
                if df_renamed[column].dtype == 'object'
                and not column.endswith('_ord')
                and not column.endswith('_target')
                ]

    # Create a list of ordinal columns (i.e. having the '_ord') suffix if any are present
    # NOTE: The ordinal columns must NOT have the '_target' suffix - they must not be target features
    ord_cols = [column for column in df_renamed.columns
                if column.endswith('_ord')
                and not column.endswith('_target')
                ]

    # Create a list of columns which are numerical, not ordinal, and do NOT have the '_target' suffix
    num_cols = [column for column in df_renamed.columns
                if pd.api.types.is_numeric_dtype(df_renamed[column])  # Use pandas' built-in numeric type checking
                and not column.endswith('_target')
                and not column.endswith('_ord')]

    # Create a list of target columns (columns with '_target' suffix)
    target_cols = [column for column in df_renamed.columns if column.endswith('_target')]

    # Print a notification of whether there are any ordinal-tagged features in the dataset
    if ord_cols:
        logger.info(f'NOTE: {len(ord_cols)} ordinal columns/features are present in the dataset.')
    else:
        logger.info('NOTE: No ordinal columns/features are tagged in the dataset.')

    # Print and log the names of the categorical features
    if cat_cols:
        logger.info('\nThe categorical non-target columns/features are:')
        logger.info(', '.join(cat_cols))
    else:
        logger.info('No categorical non-target columns/features were found in the dataset.')

    # Print and log the names of the ordinal features (if present)
    if ord_cols:
        logger.info('\nThe ordinal non-target columns/features are:')
        logger.info(', '.join(ord_cols))

    # Print and log the names of the numerical features ('The numerical non-target features are:')
    if num_cols:
        logger.info('\nThe numerical non-target columns/features are:')
        logger.info(', '.join(num_cols))
    else:
        logger.info('No numerical non-target columns/features were found in the dataset.')

    print('Producing key values at the feature level...')
    print('NOTE: Key values at the feature level are printed but not logged.')  # Notify user

    def show_key_vals(column: str, df: pd.DataFrame, feature_type: str):
        """This helper function calculates and prints key values and missingness info at the feature level."""
        print(f'\n{"-" * 50}')  # Create visual separator
        print(f'Key values for {feature_type} feature {column}:')  # Define feature
        print(f'\n{"-" * 50}')  # Visual separator

        # Calculate missingness at feature level
        missing_cnt = df[column].isnull().sum()  # Calculate total missing values
        missing_rate = (missing_cnt / len(df) * 100).round(2)
        print(f'Missingness information for {column}:')
        print(f'\n{missing_cnt} missing values - ({missing_rate}% missing)')

        # Ensure the feature is not fully null before producing value counts for categorical and ordinal features
        if not df[column].isnull().all():
            if feature_type in ['Categorical', 'Ordinal']:  # Note: these are generated and passed in the outer function
                print(f'\nValue counts:')
                print(df[column].value_counts())  # Print value counts
                # Print mode value if present - note that if multiple modes exist we produce the first mode
                print(f'Mode: {df[column].mode().iloc[0] if not df[column].mode().empty else "No mode value present"}')

            # Produce additional key stats for numerical features
            if feature_type == 'Numerical':
                print(f'\nMean: {df[column].mean():.4f}')
                print(f'Median: {df[column].median():.4f}')
                print(f'Standard deviation: {df[column].std():.4f}')
                print(f'Minimum: {df[column].min():.4f}')
                print(f'Maximum: {df[column].max():.4f}')

    # Call helper function for each feature class
    if cat_cols:  # If categorical features are present
        print('\nKEY VALUES FOR CATEGORICAL FEATURES:')
        for column in cat_cols:
            show_key_vals(column, df_renamed, 'Categorical')

        # Move to feature-class level information and log
        print('Producing key values at the feature-class level...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')

        # Build and log summary table for categorical features
        cat_summary = pd.DataFrame({
            'Unique_values': [df_renamed[column].nunique() for column in cat_cols],
            'Missing_count': [df_renamed[column].isnull().sum() for column in cat_cols],
            'Missing_rate': [(df_renamed[column].isnull().sum() / len(df_renamed) * 100).round(2)
                             for column in cat_cols]
        }, index=cat_cols)
        logger.info('\nCategorical Features Summary Table:')
        logger.info('\n' + str(cat_summary))

    if ord_cols:  # If ordinal features are present
        print('\nKEY VALUES FOR ORDINAL FEATURES:')
        for column in ord_cols:
            show_key_vals(column, df_renamed, 'Ordinal')

        # Move to feature-class level information and log
        print('Producing key values at the feature-class level...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')

        # Build and log summary table for ordinal features
        ord_summary = pd.DataFrame({
            'Unique_values': [df_renamed[column].nunique() for column in ord_cols],
            'Missing_count': [df_renamed[column].isnull().sum() for column in ord_cols],
            'Missing_rate': [(df_renamed[column].isnull().sum() / len(df_renamed) * 100).round(2)
                             for column in ord_cols]
        }, index=ord_cols)
        logger.info('\nOrdinal Features Summary Table:')
        logger.info('\n' + str(ord_summary))

    if num_cols:  # If numerical features are present
        print('\n KEY VALUES FOR NUMERICAL FEATURES:')
        for column in num_cols:
            show_key_vals(column, df_renamed, 'Numerical')

        # Move to feature-class level information and log
        print('Producing key values at the feature-class level...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')

        # Build and log summary table for numerical features
        num_summary = df_renamed[num_cols].describe()  # We can do this with a built-in Pandas method
        logger.info('\nNumerical Features Summary Table:')
        logger.info('\n' + str(num_summary))

    # Print key values for target features
    if target_cols:  # If target features are present
        print('\nTARGET FEATURE STATISTICS')
        for column in target_cols:
            # Note that we use the pandas type-assessment method to choose the string to pass to show_key_vals
            show_key_vals(column, df_renamed,
                          'Numerical' if pd.api.types.is_numeric_dtype(df_renamed[column]) else 'Categorical')

        # Build and log summary table for target features
        # We call .describe() for numerical target features and produce value counts otherwise
        logger.info('\nTarget Features Summary:')
        for column in target_cols:
            if pd.api.types.is_numeric_dtype(df_renamed[column]):
                logger.info(f'\nSummary statistics for target feature {column}:')
                logger.info(df_renamed[column].describe())

            else:
                logger.info(f'\nValue counts for target feature {column}:')
                logger.info(df_renamed[column].value_counts())

    # Return the tuple of the lists of columns by type for use in the encoding and scaling functions
    return cat_cols, ord_cols, num_cols


def impute_missing_data(df_renamed: pd.DataFrame) -> pd.DataFrame:
    """
    This function allows the user to perform simple imputation for missing values at the feature level. Three imputation
    methods are offered: mean, median, and mode imputation. The missingness rate for each feature is used to select
    features which are good candidates for imputation.
    Args:
        df_renamed (pd.DataFrame): The dataframe containing the trimmed and renamed data.
    Returns:
        df_imputed (pd.DataFrame): The dataframe after imputation is performed.
    """
    df_imputed = df_renamed.copy(deep=True)  # Create copy of the renamed dataset

    logger.info('Preparing for missing-value imputation...')  # Log update for user

    # Check if there are no missing values anywhere in the dataset
    if not df_imputed.isnull().any().any():  # Stacking calls to .any() to return a single Boolean for the series
        logger.info('No missing values present in dataset. Skipping imputation.')  # Log this
        return df_imputed  # And return the unmodified dataframe

    # Ask user if they want to perform imputation
    user_impute = input('Do you want to impute missing values? (Y/N): ')
    if user_impute.lower() != 'y':  # If the user wants to skip imputation
        logger.info('Skipping imputation.')  # Log the choice
        return df_imputed  # And return the unmodified dataframe

    # For each feature, log the count and rate of missingness
    logger.info('\nCount and rate of missingness for each feature:')
    missingness_vals = {}  # Instantiate an empty dictionary to hold the feature-level missingness values
    for column in df_imputed.columns:
        missing_cnt = df_imputed[column].isnull().sum()  # Calculate missing count
        missing_rate = (missing_cnt / len(df_imputed) * 100).round(2)  # Calculate missing rate
        missingness_vals[column] = {'count': missing_cnt, 'rate': missing_rate}  # Add those values to the dictionary
        logger.info(f'Feature {column} has {missing_cnt} missing values. ({missing_rate}% missing)')  # Log this info

    # Warn user about imputation thresholds
    print('\nWARNING: Imputing missing values for features with a missing rate over 10% is not recommended '
          'due to potential bias introduction.')

    # Build a list of good candidate features for imputation (missingness rate >0% but <=10%) using the dictionary
    imp_candidates = [key for key, value in missingness_vals.items() if 0 < value['rate'] <= 10]

    if imp_candidates:  # If any candidate features exist
        logger.info('Based on missingness rates, the following features are good candidates for imputation:')
        for key in imp_candidates:
            logger.info(f'- {key}: {missingness_vals[key]["rate"]}% missing')

    else:  # If no good candidate features are present
        logger.info('No features fall within the recommended rate range for imputation.')  # Log that fact
        print('WARNING: Statistical best practices indicate you should not perform imputation.')  # Print user warning

    # Ask if user wants to override the recommendation
    while True:  # We can justify 'while True' because we have a cancel-out input option
        try:
            user_override = input('\nDo you wish to:\n'
                                  '1. Impute only for recommended features (<= 10% missing)\n'
                                  '2. Override the warning and consider all features with missing values\n'
                                  '3. Skip imputation').strip()

            # Validate input
            if user_override not in ['1', '2', '3']:
                raise ValueError('Please enter 1, 2, or 3.')

            # Check for cancellation
            if user_override.lower() == '3':
                logger.info('Skipping imputation. No changes made to dataset.')
                return df_imputed

            # Build list of features to be imputed
            imp_features = (imp_candidates if user_override == '1'
                            else [key for key, value in missingness_vals.items() if value['count'] > 0])

            # Validate that the list of features for imputation isn't empty
            if not imp_features:
                logger.info('No features available for imputation given user input. Skipping imputation.')  # Log this
                return df_imputed  # And return the unmodified dataset

            break  # Exit the while loop if we get valid user input

        except ValueError as exc:  # Catch invalid input
            print(f'Invalid input: {exc}')
            continue  # And restart the while loop

    # Print warning that TADPREPS only supports simple imputation methods
    print('\nWARNING: TADPREPS supports only mean, median, and mode imputation.')
    print('For more sophisticated methods (e.g. imputation-by-modeling), skip this step and write '
          'your own imputation code.')

    # Allow the user to exit the process if they don't want to use simple imputation methods
    user_proceed = input('Do you want to proceed using simple imputation methods? (Y/N): ')
    if user_proceed.lower() != 'y':  # If the user wants to skip imputation
        logger.info('Skipping imputation. No changes made to dataset.')  # Log this
        return df_imputed  # And return the unmodified dataset

    # Ask the user if they want a refresher on the three imputation methods offered
    user_refresh = input('Do you want to see a brief refresher on these imputation methods? (Y/N): ')
    if user_refresh.lower() == 'y':
        print('\nSimple Imputation Methods Overview:'
              '\n- Mean: Best for normally-distributed data. Theoretically practical. Highly sensitive to outliers.'
              '\n- Median: Better for skewed numerical data. Robust to outliers.'
              '\n- Mode: Useful for categorical and fully-discrete numerical data.')

    # Begin imputation at feature level
    for feature in imp_features:
        print(f'\nProcessing feature {feature}...')
        print(f'- Datatype: {df_imputed[feature].dtype}')
        print(f'- Missing rate: {missingness_vals[feature]["rate"]}%')

        # Build list of available/valid imputation methods based on feature datatype
        if pd.api.types.is_numeric_dtype(df_imputed[feature]):  # For numerical features
            val_methods = ['Mean', 'Median', 'Mode', 'Skip imputation for this feature']
        else:
            val_methods = ['Mode', 'Skip imputation for this feature']  # Only mode is valid for non-numeric data

        # Prompt user to select an imputation method
        while True:
            method_items = [f'{idx}. {method}' for idx, method in enumerate(val_methods, 1)]
            method_prompt = f'Choose imputation method:\n{"\n".join(method_items)}\nEnter the number of your choice: '
            user_imp_choice = input(method_prompt)

            try:
                method_idx = int(user_imp_choice) - 1  # Find correct integer value for index of selected method

                if 0 <= method_idx < len(val_methods):  # Validate input
                    imp_method = val_methods[method_idx]  # Select imputation method
                    break  # And exit the while loop
                else:  # Catch invalid integers
                    print('Invalid input. Enter a valid number.')

            # Catch other input errors
            except ValueError:
                print('Invalid input. Enter a valid number.')

        if imp_method == 'Skip imputation for this feature':  # If user wants to skip a feature
            logger.info(f'Skipping imputation for feature: {feature}')  # Log the choice
            continue  # And restart the outer for loop with the next feature

        # Begin actual imputation process
        try:
            # Calculate impute values based on method selection
            if imp_method == 'Mean':
                imp_val = df_imputed[feature].mean()
            elif imp_method == 'Median':
                imp_val = df_imputed[feature].median()
            else:  # If only mode is a valid method
                mode_vals = df_imputed[feature].mode()
                if len(mode_vals) == 0:  # If no mode values exist
                    # Log a warning
                    logger.warning(f'No mode value exists for feature {feature}. Skipping imputation for this feature.')
                    continue  # Restart outer for loop with next feature
                imp_val = mode_vals[0]  # Select first mode

            # Impute missing values at feature level
            feature_missing_cnt = missingness_vals[feature]['count']
            logger.info(f'Replacing {feature_missing_cnt} missing values for {feature} '
                        f'using {imp_method} value of {imp_val}.')
            df_imputed[feature].fillna(imp_val, inplace=True)  # Replace empty values with imputed value in-place

        # Catch all other exceptions
        except Exception as exc:
            # Log errors
            logger.error(f'Error during imputation for feature {feature}: {exc}')
            logger.error('Skipping imputation for this feature.')
            continue  # Restart outer for loop with next feature

    return df_imputed  # Return the new dataframe with imputed values


def encode_features(df_imputed: pd.DataFrame, cat_cols: list, ord_cols: list) -> pd.DataFrame:
    """
    This function allows the user to use appropriate encoding methods on the categorical and ordinal non-target
    features in the dataset. It identifies the appropriate candidate features using the lists created by the
    print_feature_stats function.
    Args:
        df_imputed (pd.Dataframe): The dataframe with imputed values return by impute_missing) data.
        cat_cols (list): The list of categorical non-target features created by print_feature_stats().
        ord_cols (list): The list of ordinal non-target features created by print_feature_stats().
    Returns:
        df_encoded (pd.DataFrame): A dataframe with encoded categorical and ordinal features.
    """
    df_encoded = df_imputed.copy(deep=True)  # Create copy of imputed dataframe

    if not cat_cols and not ord_cols:  # If there are no candidate features for recoding
        logger.info('No categorical or ordinal features are present in the dataset. Skipping encoding.')  # Log this
        return df_encoded  # And return the unmodified dataset

    # Log feature count info
    logger.info(f'The dataset contains {len(cat_cols)} categorical and {len(ord_cols)} ordinal non-target features.')

    # Ask if user wants to perform any encoding
    user_encode = input('Do you wish to encode any of these features? (Y/N): ')
    if user_encode.lower() != 'y':  # If user does not wish to encode
        logger.info('Skipping encoding.')  # Log this
        return df_encoded  # And return the unmodified dataset

    if cat_cols:
        print('Preparing to encode categorical features...')

    # Ask if the user wants to see a brief refresher on one-hot vs. dummy encoding for categorical features
    # If so, display the refresher

    '''
    Now, for each categorical feature (assuming any are present - check for this):
    - Ask if the user wants to encode this feature. If not, skip to next feature
    - Print total # of unique categories/values in the feature
    - Print alphabetized unique list of categories/values in the feature
    - Print warning message that if the number of categories is too high, (i.e. greater than about 20) the user should perform some kind of dimensionality reduction rather than encode the feature
    - Note that this helps avoid the curse of dimensionality and overly-sparse data in the features 
    - Print value counts for the feature
    - Ask if the user wants to see a barplot of the feature distribution
    - Ask if the user wants to one-hot encode or dummy the feature
    - Apply selected encoding method (or none, if user doesn't want to encode) to the feature
    '''

    if ord_cols:
        print('Preparing to encode ordinal features...')

    # Ask if the user wants to see a brief refresher on ordinal encoding for ordinal features
    # If so, display the refresher

    '''
    Now, for each ordinal feature (assuming any are present - check for this):
    - Ask if the user wants to encode this feature. If not, skip to next feature
    - Print total # of unique categories/values in the feature
    - Print alphabetized unique list of categories/values in the feature
    - Print warning message that if the number of categories is too high, (i.e. greater than about 20) the user should perform some kind of dimensionality reduction rather than encode the feature
    - Note that this helps avoid the curse of dimensionality and overly-sparse data in the features 
    - Print value counts for the feature
    - Ask if the user wants to see a barplot of the feature distribution
    - Ask if the user wants to encode the ordinal feature
    - Apply ordinal encoding (or none, if user doesn't want to encode) to the feature
    '''

    return df_encoded  # Return the dataset with encoding applied
