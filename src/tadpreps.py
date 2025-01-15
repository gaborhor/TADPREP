"""
This is the TADPREPS codebase. All core functions will be defined internally (so no additional files are necessary at
runtime) and the logging file will be created in the same working directory as the script itself.
"""

# Library imports
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import shutil
import sys
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Fetch current runtime timestamp in a readable format
timestamp = datetime.now().strftime('%Y%m%d_%H%M')

# Set up first-phase temporary log
temp_log_name = f'tadpreps_runtime_log_{timestamp}.log'
temp_log_path = Path(temp_log_name)

# Set up logging with time-at-execution
# We persist with "%-type" formatting to preserve backward compatibility
logging.basicConfig(
    level=logging.INFO, format='%(levelname)s: %(message)s',
    handlers=[logging.FileHandler(temp_log_path), logging.StreamHandler(sys.stdout)]
)

# Instantiate log object
logger = logging.getLogger(__name__)
logger.info('Initiating TADPREPS...')


def relocate_log(target_dir: Path) -> None:
    """
    This allows for the second phase of logging - i.e. co-locating the log on disk with the exported file.
    Relocates log file to the target directory supplied by the user in export_file() and updates logging configuration.
    Args:
        target_dir (Path): Path object representing the target directory
    """
    try:
        # Store the current handlers
        handlers = logger.handlers[:]

        # Remove and close all current handlers
        for handler in handlers:
            # Flush and close the handler properly
            handler.flush()
            handler.close()
            logger.removeHandler(handler)

        # Now that handlers are closed, move the file
        new_log_path = target_dir / temp_log_path.name
        shutil.move(temp_log_path, new_log_path)

        # Create a new file handler for the new location
        new_handler = logging.FileHandler(new_log_path)
        new_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(new_handler)

        # Recreate the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(console_handler)

        logger.info(f'Log file relocated to: {new_log_path}')

    # Catch relocation errors
    except Exception as error:
        logger.error(f'Failed to relocate log file: {error}')
        logger.error('Continuing with original log location.')

        # If relocation failed, restore original handlers
        if not logger.handlers:
            file_handler = logging.FileHandler(temp_log_path)
            file_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            logger.addHandler(console_handler)


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
    # Print number of instances in file
    logger.info(f'The unaltered file has {df_full.shape[0]} instances.')  # [0] is rows

    # Print number of features in file
    logger.info(f'The unaltered file has {df_full.shape[1]} features.')  # [1] is columns
    print('-' * 50)  # Visual separator

    # Print names and datatypes of features in file
    logger.info('Names and datatypes of features:')
    logger.info(df_full.info(memory_usage=False, show_counts=False))  # Limit information printed since it's logged
    print('-' * 50)  # Visual separator

    # Print # of instances with missing values
    row_missing_cnt = df_full.isnull().any(axis=1).sum()  # Compute count
    row_missing_rate = (row_missing_cnt / len(df_full) * 100).round(2)  # Compute rate
    logger.info(f'{row_missing_cnt} instances ({row_missing_rate}%) contain at least one missing value.')


def trim_file(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    This function allows the user to delete instances with any missing values, to drop features from the
    dataset, and to sub-set the data by deleting a specified proportion of the instances at random.
    Args:
        df_full (pd.DataFrame): The original, unaltered dataset.
    Returns:
        df_trimmed (pd.DataFrame): The dataset after trimming/sub-setting.
    """
    df_trimmed = df_full.copy(deep=True)
    # Ask if the user wants to delete *all* instances with any missing values
    user_drop_na = input('Do you want to drop all instances with *any* missing values? (Y/N): ')
    if user_drop_na.lower() == 'y':
        df_trimmed = df_trimmed.dropna()
        logger.info(f'After deletion of instances with missing values, {len(df_trimmed)} instances remain.')

    # Ask if the user wants to drop any of the columns/features in the dataset
    user_drop_cols = input('Do you want to drop any of the features in the dataset? (Y/N): ')
    if user_drop_cols.lower() == 'y':
        print('The full set of features in the dataset is:')
        for col_idx, column in enumerate(df_trimmed.columns, 1):  # Create enumerated list of features starting at 1
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                drop_cols_input = input('\nEnter the index integers of the features you wish to drop '
                                        '(comma-separated) or enter "C" to cancel: ')

                # Check for user cancellation
                if drop_cols_input.lower() == 'c':  # If user entered cancel-out input
                    logger.info('Feature deletion cancelled.')  # Log the cancellation
                    break  # Exit the while loop

                # Create list of column indices to drop
                drop_cols_idx = [int(idx.strip()) for idx in drop_cols_input.split(',')]  # Splitting on comma

                # Verify that all index numbers of columns to be dropped are valid/in range
                if not all(1 <= idx <= len(df_trimmed.columns) for idx in drop_cols_idx):  # Using a generator
                    raise ValueError('Some feature index integers entered are out of range/invalid.')

                # Convert specified column numbers to actual column names
                drop_cols_names = [df_trimmed.columns[idx - 1] for idx in drop_cols_idx]  # Subtracting 1 from indices

                # Drop the columns
                df_trimmed.drop(columns=drop_cols_names, inplace=True)
                logger.info(f'Dropped features: {",".join(drop_cols_names)}')  # Log the dropped columns
                break  # Exit the while loop

            # Catch invalid user input
            except ValueError:
                logger.error('Invalid input. Please enter valid feature index integers separated by commas.')
                continue  # Restart the loop

    # Ask if the user wants to sub-set the data by deleting a specified proportion of the instances at random
    user_subset = input('Do you want to sub-set the data by randomly deleting a specified proportion of '
                        'instances? (Y/N): ')
    if user_subset.lower() == 'y':
        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or '
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
                    logger.info(f'Randomly dropped {subset_rate}% of instances. {retain_row_cnt} instances '
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
    This function allows the user to rename features and to append the '_ord' and/or '_target' suffixes
    to ordinal or target columns.
    Args:
        df_trimmed (pd.DataFrame): The 'trimmed' dataset created by trim_file().
    Returns:
        df_renamed (pd.DataFrame): The dataset with renamed columns.
    """
    df_renamed = df_trimmed.copy(deep=True)  # Create copy of trimmed dataset

    # Ask if user wants to rename any columns
    user_rename_cols = input('Do you want to rename any of the features in the dataset? (Y/N): ')
    if user_rename_cols.lower() == 'y':
        print('The list of features currently present in the dataset is:')
        for col_idx, column in enumerate(df_renamed.columns, 1):  # Create enumerated list of features starting at 1
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                rename_cols_input = input('\nEnter the index integer of the feature you wish to rename '
                                          'or enter "C" to cancel: ')

                # Check for user cancellation
                if rename_cols_input.lower() == 'c':  # If user entered cancel-out input
                    logger.info('Feature renaming cancelled.')  # Log the cancellation
                    break  # Exit the while loop

                col_idx = int(rename_cols_input)  # Convert input to integer
                if not 1 <= col_idx <= len(df_renamed.columns):  # Validate entry
                    raise ValueError('Column index is out of range.')

                # Get new name for the column from user
                col_name_old = df_renamed.columns[col_idx - 1]
                col_name_new = input(f'Enter new name for feature "{col_name_old}": ').strip()

                # Validate name to make sure it doesn't already exist in the dataset
                if col_name_new in df_renamed.columns:
                    logger.error(f'Feature name "{col_name_new}" already exists. Choose a different name.')
                    continue  # Restart the loop

                # Rename column in-place
                df_renamed.rename(columns={col_name_old: col_name_new}, inplace=True)
                logger.info(f'Renamed feature "{col_name_old}" to "{col_name_new}".')

                # Ask if user wants to rename another column
                if input('Do you want to rename another feature? (Y/N): ').lower() != 'y':
                    break  # If not, exit the while loop

            except ValueError as exc:
                logger.error(f'Invalid input: {exc}')

    # Ask if user wants to perform batch-level appending of '_ord' tag to ordinal features
    user_tag_ord = input('Do you want to tag any features as ordinal by appending the "_ord" suffix '
                         'to their names? (Y/N): ')

    if user_tag_ord.lower() == 'y':
        print('\nCurrent feature list:')
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
                    raise ValueError('Some feature indices are out of range.')

                ord_names_pretag = [df_renamed.columns[idx - 1] for idx in ord_idx_list]  # Create list of pretag names

                # Generate mapper for renaming columns with '_ord' suffix
                ord_rename_map = {name: f'{name}_ord' for name in ord_names_pretag if not name.endswith('_ord')}

                # Validate that tags for the selected columns are not somehow already present (i.e. done pre-import)
                if not ord_rename_map:  # If the mapper is empty
                    logger.warning('All selected features are already tagged as ordinal.')  # Warn the user
                    break  # And exit the while loop

                df_renamed.rename(columns=ord_rename_map, inplace=True)  # Perform tagging
                logger.info(f'Tagged the following features as ordinal: {", ".join(ord_rename_map.keys())}')
                break

            # Catch invalid input
            except ValueError as exc:
                logger.error(f'Invalid input: {exc}')
                continue  # Restart the loop

    # Ask if user wants to perform batch-level appending of '_target' tag to target features
    user_tag_target = input('Do you want to tag any features as targets by appending the "_target" suffix '
                            'to their names? (Y/N): ')

    if user_tag_target.lower() == 'y':
        print('\nCurrent features:')
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
                    raise ValueError('Some feature indices are out of range.')

                target_names_pretag = [df_renamed.columns[idx - 1] for idx in target_idx_list]  # List of pretag names

                # Generate mapper for renaming columns with '_target' suffix
                target_rename_map = {name: f'{name}_target' for name in target_names_pretag if
                                     not name.endswith('_target')}

                # Validate that tags for the selected columns are not somehow already present (i.e. done pre-import)
                if not target_rename_map:  # If the mapper is empty
                    logger.warning('All selected features are already tagged as targets.')  # Warn the user
                    break  # And exit the while loop

                df_renamed.rename(columns=target_rename_map, inplace=True)  # Perform tagging
                logger.info(f'Tagged the following features as targets: {", ".join(target_rename_map.keys())}')
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
    logger.info('Displaying top-level information for features in dataset...')

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

    def handle_numeric_cats(df: pd.DataFrame, init_num_cols: list[str]) -> tuple[list[str], list[str]]:
        """
        This internal helper function identifies numerical features that might actually be categorical in function.
        If such a feature is identified as categorical, the function converts its values to strings in-place.
        It returns two lists: one of confirmed numerical columns and another for newly-identified categorical columns.
        """
        # Instantiate empty lists to hold the features after parsing
        true_num_cols = []
        true_cat_cols = []

        for column in init_num_cols:  # For each feature that was initially identified as numeric
            unique_vals = sorted(df[column].unique())  # Calculate number of unique values
            if (len(unique_vals) <= 5 and  # If that value is small
                    all(float(x).is_integer() for x in unique_vals if pd.notna(x))):  # And all values are integers

                print(f'\nFeature "{column}" has {len(unique_vals)} unique values: {unique_vals}')
                print('ALERT: This could be a categorical feature encoded as numbers.')
                print('E.g., this might be similar to 1/0 encoding for Yes/No responses.')

                user_cat = input(f'Should "{column}" actually be treated as categorical? (Y/N): ')  # Ask user to choose
                if user_cat.lower() == 'y':  # If so
                    df[column] = df[column].astype(str)  # Cast the values to strings in-place
                    true_cat_cols.append(column)  # And add the identified feature to the true_cat_cols list
                    logger.info(f'Converted feature "{column}" to categorical type.')  # Log the choice

                # Otherwise, if user says no, treat the value as truly numeric
                else:
                    true_num_cols.append(column)

            # If the feature fails the checks, treat it as truly numeric
            else:
                true_num_cols.append(column)

        return true_num_cols, true_cat_cols  # Return the lists of true numerical and identified categorical features

    # Call the helper function
    num_cols, new_cat_cols = handle_numeric_cats(df_renamed, num_cols)
    cat_cols.extend(new_cat_cols)  # Add any new identified categorical features to cat_cols

    # Print a notification of whether there are any ordinal-tagged features in the dataset
    if ord_cols:
        logger.info(f'NOTE: {len(ord_cols)} ordinal features are present in the dataset.')
        print('-' * 50)  # Visual separator
    else:
        logger.info('NOTE: No ordinal features are tagged in the dataset.')
        print('-' * 50)  # Visual separator

    # Print and log the names of the categorical features
    if cat_cols:
        logger.info('The categorical non-target features are:')
        logger.info(', '.join(cat_cols))
        print('-' * 50)  # Visual separator
    else:
        logger.info('No categorical non-target features were found in the dataset.')
        print('-' * 50)  # Visual separator

    # Print and log the names of the ordinal features (if present)
    if ord_cols:
        logger.info('The ordinal non-target features are:')
        logger.info(', '.join(ord_cols))
        print('-' * 50)  # Visual separator

    # Print and log the names of the numerical features ('The numerical non-target features are:')
    if num_cols:
        logger.info('The numerical non-target features are:')
        logger.info(', '.join(num_cols))
        print('-' * 50)  # Visual separator
    else:
        logger.info('No numerical non-target features were found in the dataset.')
        print('-' * 50)  # Visual separator

    print('Producing key values at the feature level...')
    print('NOTE: Key values at the feature level are printed but not logged.')  # Notify user

    def show_key_vals(column: str, df: pd.DataFrame, feature_type: str):
        """This helper function calculates and prints key values and missingness info at the feature level."""
        print('-' * 50)  # Visual separator
        print(f'Key values for {feature_type} feature "{column}":')  # Define feature
        print('-' * 50)  # Visual separator

        # Calculate missingness at feature level
        missing_cnt = df[column].isnull().sum()  # Calculate total missing values
        missing_rate = (missing_cnt / len(df) * 100).round(2)
        print(f'Missingness information for "{column}":')
        print(f'{missing_cnt} missing values - ({missing_rate}% missing)')

        # Ensure the feature is not fully null before producing value counts for categorical and ordinal features
        if not df[column].isnull().all():
            if feature_type in ['Categorical', 'Ordinal']:  # Note: these are generated and passed in the outer function
                print(f'\nValue counts:')
                print(df[column].value_counts())  # Print value counts
                # Print mode value if present - note that if multiple modes exist we produce the first mode
                print(f'Mode: {df[column].mode().iloc[0] if not df[column].mode().empty else "No mode value exists."}')

            # Produce additional key stats for numerical features
            if feature_type == 'Numerical':
                print(f'\nMean: {df[column].mean():.4f}')
                print(f'Median: {df[column].median():.4f}')
                print(f'Standard deviation: {df[column].std():.4f}')
                print(f'Minimum: {df[column].min():.4f}')
                print(f'Maximum: {df[column].max():.4f}')

    # Call helper function for each feature class
    if cat_cols:  # If categorical features are present
        print('-' * 50)  # Visual separator
        print('KEY VALUES FOR CATEGORICAL FEATURES:')
        for column in cat_cols:
            show_key_vals(column, df_renamed, 'Categorical')

        # Move to feature-class level information and log
        print('-' * 50)  # Visual separator
        print('Producing key values at the feature-class level...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')
        print('-' * 50)  # Visual separator

        # Build and log summary table for categorical features
        cat_summary = pd.DataFrame({
            'Unique_values': [df_renamed[column].nunique() for column in cat_cols],
            'Missing_count': [df_renamed[column].isnull().sum() for column in cat_cols],
            'Missing_rate': [(df_renamed[column].isnull().sum() / len(df_renamed) * 100).round(2)
                             for column in cat_cols]
        }, index=cat_cols)
        logger.info('Categorical Features Summary Table:')
        logger.info(str(cat_summary))

    if ord_cols:  # If ordinal features are present
        print('-' * 50)  # Visual separator
        print('KEY VALUES FOR ORDINAL FEATURES:')
        for column in ord_cols:
            show_key_vals(column, df_renamed, 'Ordinal')

        # Move to feature-class level information and log
        print('-' * 50)  # Visual separator
        print('Producing key values at the feature-class level...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')
        print('-' * 50)  # Visual separator

        # Build and log summary table for ordinal features
        ord_summary = pd.DataFrame({
            'Unique_values': [df_renamed[column].nunique() for column in ord_cols],
            'Missing_count': [df_renamed[column].isnull().sum() for column in ord_cols],
            'Missing_rate': [(df_renamed[column].isnull().sum() / len(df_renamed) * 100).round(2)
                             for column in ord_cols]
        }, index=ord_cols)
        logger.info('Ordinal Features Summary Table:')
        logger.info(str(ord_summary))

    if num_cols:  # If numerical features are present
        print('-' * 50)  # Visual separator
        print('KEY VALUES FOR NUMERICAL FEATURES:')
        for column in num_cols:
            show_key_vals(column, df_renamed, 'Numerical')

        # Move to feature-class level information and log
        print('-' * 50)  # Visual separator
        print('Producing key values at the feature-class level...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')
        print('-' * 50)  # Visual separator

        # Build and log summary table for numerical features
        num_summary = df_renamed[num_cols].describe()  # We can do this with a built-in Pandas method
        logger.info('Numerical Features Summary Table:')
        logger.info(str(num_summary))

    # Print key values for target features
    if target_cols:  # If target features are present
        print('-' * 50)  # Visual separator
        print('TARGET FEATURE STATISTICS')
        for column in target_cols:
            # Note that we use the pandas type-assessment method to choose the string to pass to show_key_vals
            show_key_vals(column, df_renamed,
                          'Numerical' if pd.api.types.is_numeric_dtype(df_renamed[column]) else 'Categorical')

        # Build and log summary table for target features
        # We call .describe() for numerical target features and produce value counts otherwise
        print('-' * 50)  # Visual separator
        logger.info('Target Features Summary Table:')
        for column in target_cols:
            if pd.api.types.is_numeric_dtype(df_renamed[column]):
                logger.info(f'Summary statistics for target feature {column}:')
                logger.info(df_renamed[column].describe())

            else:
                logger.info(f'Value counts for target feature {column}:')
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

    imp_features = []  # Instantiate list of features for imputation before the try block begins (assignment resilience)

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
    print('-' * 50)  # Visual separator
    logger.info('Count and rate of missingness for each feature:')
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
                                  '3. Skip imputation\n'
                                  'Enter choice: ').strip()

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
    user_imp_proceed = input('Do you want to proceed using simple imputation methods? (Y/N): ')
    if user_imp_proceed.lower() != 'y':  # If the user wants to skip imputation
        logger.info('Skipping imputation. No changes made to dataset.')  # Log this
        return df_imputed  # And return the unmodified dataset

    # Ask the user if they want a refresher on the three imputation methods offered
    user_impute_refresh = input('Do you want to see a brief refresher on these imputation methods? (Y/N): ')
    if user_impute_refresh.lower() == 'y':
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
            df_imputed = df_imputed.fillna({feature: imp_val})  # Replace empty values with imputed value in-place

        # Catch all other exceptions
        except Exception as exc:
            # Log errors
            logger.error(f'Error during imputation for feature {feature}: {exc}')
            logger.error('Skipping imputation for this feature.')
            continue  # Restart outer for loop with next feature

    return df_imputed  # Return the new dataframe with imputed values


def encode_and_scale(df_imputed: pd.DataFrame, cat_cols: list[str], ord_cols: list[str], num_cols: list[str]) \
        -> pd.DataFrame:
    """
    This function allows the user to use appropriate encoding methods on the categorical and ordinal non-target
    features in the dataset. It identifies the appropriate candidate features using the lists created by the
    print_feature_stats function. Note that the actual 'engine' of this function is the set of three helper functions
    defined within its scope.
    Args:
        df_imputed (pd.Dataframe): The dataframe with imputed values return by impute_missing) data.
        cat_cols (list): The list of categorical non-target features created by print_feature_stats().
        ord_cols (list): The list of ordinal non-target features created by print_feature_stats().
        num_cols (list): The list of numerical non-target features created by print_feature_stats().
    Returns:
        df_final (pd.DataFrame): A final-form dataframe with encoded and scaled features.
    """
    df_final = df_imputed.copy(deep=True)  # Create copy of imputed dataframe

    def handle_cats():
        """This internal helper function facilitates the encoding of categorical features, if desired by the user."""
        nonlocal df_final  # Specify that this object comes from the outer scope

        if not cat_cols:  # If the list of categorical columns is empty
            logger.info('No categorical features are present in the dataset. Skipping encoding.')  # Log this
            return  # And exit the process

        print('-' * 50)  # Visual separator
        logger.info(f'The dataset contains {len(cat_cols)} categorical features.')  # Print # of categorical features

        # Notify user that TADPREPS only supports common encoding methods
        print('\nNOTE: TADPREPS only supports One-Hot and Dummy encoding.')
        print('If you wish to use other, more complex encoding methods, skip this step and write your own code.')

        # Ask if user wants to proceed
        user_encode = input('\nDo you want to Dummy or One-Hot encode any of the categorical features? (Y/N): ')
        if user_encode.lower() != 'y':  # If not
            logger.info('Skipping encoding of categorical features.')  # Log that fact
            return  # And exit the process

        # Ask if user wants a refresher on One-Hot vs. Dummy encoding
        user_encode_refresh = input('Do you want to see a brief refresher on One-Hot vs. Dummy encoding? (Y/N): ')
        if user_encode_refresh.lower() == 'y':  # If so, display refresher
            print('\nOverview of One-Hot vs. Dummy Encoding:'
                  '\nOne-Hot Encoding: '
                  '\n- Creates a new binary column for every unique category. '
                  '\n- No information is lost, which preserves interpretability, but more features are created.'
                  '\n- This method is preferred for non-linear models like decision trees.'
                  '\n'
                  '\nDummy Encoding:'
                  '\n- Creates n-1 binary columns for n categories in the feature. '
                  '\n- One category becomes the reference and is represented by all zeros. '
                  '\n- Dummy encoding is preferred for linear models to avoid perfect multicollinearity.'
                  '\n- This method is more computationally- and space-efficient but is less interpretable.')

        encoded_cols = []  # Instantiate an empty list to hold the encoded columns for final reporting
        encoded_dfs = []  # Instantiate an empty list to collect encoded DataFrames
        columns_to_drop = []  # Instantiate an empty list to collect the original columns to drop

        # Begin encoding process at the feature level
        for column in cat_cols:
            print(f'\nProcessing feature: {column}')

            # Check if user wants to encode this feature
            user_encode_feature = input(f'Do you want to encode {column}? (Y/N): ')
            if user_encode_feature.lower() != 'y':  # If not
                logger.info(f'Skipping encoding for feature: {column}')  # Log that choice
                continue  # And move to the next feature

            # Show unique value count
            unique_cnt = df_final[column].nunique()
            print(f'Feature {column} has {unique_cnt} unique values.')

            # Check for nulls before proceeding
            null_cnt = df_final[column].isnull().sum()
            if null_cnt > 0:  # If nulls are present
                # Log a warning
                logger.warning(
                    f'Feature {column} contains {null_cnt} null values. Encoding may produce unexpected results.')
                # Ask if the user wants to proceed anyway
                user_proceed = input('Do you want to proceed with encoding this feature? (Y/N): ')
                if user_proceed.lower() != 'y':  # If not
                    logger.info(f'Skipping encoding for feature: {column}')  # Log the choice
                    continue  # Move on to the next feature

            # Check for single-value features and log a warning if this feature is single-value
            if unique_cnt == 1:
                logger.warning(f'Feature {column} has only one unique value. Consider dropping this feature.')
                continue  # Move on to the next feature

            # Check for low-frequency categories
            value_counts = df_final[column].value_counts()
            low_freq_cats = value_counts[value_counts < 10]  # Using 10 as minimum category size
            if not low_freq_cats.empty:
                print(f'\nWARNING: Found {len(low_freq_cats)} categories with fewer than 10 instances:')
                print(low_freq_cats)
                print('Consider grouping rare categories before encoding.')

                user_proceed = input('Do you want to proceed with encoding despite low-frequency categories? (Y/N): ')
                if user_proceed.lower() != 'y':
                    logger.info(f'Skipping encoding for feature {column} due to low-frequency categories.')
                    continue

            # Display warning if the unique value count is too high
            if unique_cnt > 20:
                print(f'\nWARNING: Feature {column} contains more than 20 unique values.')
                print('Consider using remapping or other dimensionality reduction techniques instead of encoding.')
                print('Encoding high-cardinality features can create the curse of dimensionality.')
                print('It can also result in having overly-sparse data in your feature set.')

                # Ask if user wants to proceed despite this warning
                user_proceed = input(f'\n Do you still want to encode {column} despite this warning? (Y/N): ')
                if user_proceed.lower() != 'y':  # If not
                    logger.info(f'Skipping encoding for high-cardinality feature {column}.')  # Log the choice
                    continue  # And move to the next feature

            # Display (do not log) unique values
            print(f'\nUnique values (alphabetized) for {column}:')
            unique_vals = df_final[column].unique()

            # Filter out None and NaN values for sorting
            non_null_vals = [value for value in unique_vals if pd.notna(value)]
            sorted_vals = sorted(non_null_vals)

            # Add None/NaN values back if they existed in the original data
            if df_final[column].isnull().any():
                sorted_vals.append(None)
            for value in sorted_vals:
                if pd.isna(value):
                    print(f'- Missing/NaN')
                else:
                    print(f'- {value}')

            # Display (do not log) value counts
            print(f'\nValue counts for {column}:')
            print(df_final[column].value_counts())

            # Ask if user wants to see a distribution plot
            user_show_plot = input('\nWould you like to see a plot of the feature distribution? (Y/N): ')
            if user_show_plot.lower() == 'y':  # If so, send call to Seaborn
                plt.figure(figsize=(12, 8))
                sns.displot(data=df_final, x=column, discrete=True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()

            # Ask user to select encoding method
            while True:  # We can justify 'while True' because we have a cancel-out input option
                try:
                    print(f'\nSelect encoding method for feature {column}:')
                    print('1. One-Hot Encoding')
                    print('2. Dummy Encoding')
                    print('3. Skip encoding for this feature')
                    enc_method = input('Enter your choice (1, 2, or 3): ')

                    if enc_method == '3':  # If user wants to skip
                        logger.info(f'Skipping encoding for feature {column}.')  # Log that choice
                        break  # Exit the while loop

                    elif enc_method in ['1', '2']:  # If a valid encoding choice is made

                        if enc_method == '1':  # Perform one-hot encoding
                            # Create binary columns for each unique value in the feature
                            encoded = pd.get_dummies(df_final[column], prefix=column, prefix_sep='_')

                            # Store encoded DataFrame for later use
                            encoded_dfs.append(encoded)

                            # Store original column name so we can drop all original columns at once
                            columns_to_drop.append(column)

                            # Track this column for reporting purposes
                            encoded_cols.append(f'{column} (One-Hot)')

                            # Log the action
                            logger.info(f'Applied one-hot encoding to feature {column}.')

                        else:  # If enc_method is '2', perform dummy encoding
                            # Create n-1 binary columns, dropping first category as reference
                            encoded = pd.get_dummies(df_final[column], prefix=column, prefix_sep='_', drop_first=True)

                            # Store encoded DataFrame for later use
                            encoded_dfs.append(encoded)

                            # Store original column name so we can drop all original columns at once
                            columns_to_drop.append(column)

                            # Track this column for reporting purposes
                            encoded_cols.append(f'{column} (Dummy)')

                            # Log the action
                            logger.info(f'Applied dummy encoding to feature {column}.')

                        break  # Exit the while loop

                    else:  # Catch invalid input
                        print('Invalid input. Please enter 1, 2, or 3.')
                        continue  # Restart the while loop

                # Catch other errors
                except Exception as exc:
                    logger.error(f'Error during encoding of feature {column}: {exc}')
                    print('An error occurred. Skipping encoding for this feature.')
                    break  # Exit the while loop

        # Perform single concatenation if any encodings were done
        if encoded_dfs:
            df_final = df_final.drop(columns=columns_to_drop)
            df_final = pd.concat([df_final] + encoded_dfs, axis=1)

        # After all features are processed, log a summary of encoding results
        # If any features were encoded, log the encoded features
        if encoded_cols:
            logger.info('\nThe following features were encoded:')
            for column in encoded_cols:
                logger.info(f'- {column}')

        # If no features were encoded, log that fact
        else:
            logger.info('No features were encoded.')

    def handle_ords():
        """This internal helper function facilitates the remapping of ordinal features, if desired by the user."""
        nonlocal df_final  # Specify that this object comes from the outer scope

        if not ord_cols:  # If no columns are tagged as ordinal
            logger.info('No ordinal features are present in the dataset. Skipping remapping.')  # Log this
            return  # And exit the process

        print('-' * 50)  # Visual separator
        logger.info(f'The dataset contains {len(ord_cols)} ordinal features.')

        # Create list to track which features get remapped for final reporting
        remapped_cols = []

        # Create list of string-type ordinal features using Pandas' data-type methods
        str_ords = [column for column in ord_cols if not pd.api.types.is_numeric_dtype(df_final[column])]

        # If all ordinal features are already numerical, they don't need remapping
        if not str_ords:  # If there are no string-type ordinal features
            logger.info('All ordinal features are already in a numerical format.')  # Log that fact

            # Print (do not log) a notification/explanation for the user
            print('\nNOTE: Ordinal features in numerical format do not need to be scaled.')
            print('Scaling ordinal features distorts the meaningful distances between values.')
            logger.info('Skipping remapping of ordinal features.')  # Log the auto-skip for the entire process
            return  # Return the unmodified data

        # If there are string-type ordinal features, ask if user wants to remap them with numerical values
        logger.info(f'{len(str_ords)} ordinal features contain non-numeric values.')
        user_remap = input('\nDo you want to remap any string-type ordinal features to numerical values? (Y/N): ')
        if user_remap.lower() != 'y':  # If not
            logger.info('Skipping remapping of ordinal features.')  # Log the choice
            return  # Return the unmodified data

        # Process each string-type ordinal feature
        for column in str_ords:
            print(f'\nProcessing feature: {column}')

            # Ask if user wants to remap this specific feature
            user_remap_feature = input(f'Do you want to remap {column} to numerical values? (Y/N): ')
            if user_remap_feature.lower() != 'y':  # If not
                logger.info(f'Skipping remapping for feature: {column}')  # Log the choice
                continue  # Return the unmodified data

            # Check for nulls before proceeding
            null_cnt = df_final[column].isnull().sum()
            # If nulls are present, log a warning and ask if user wants to proceed
            if null_cnt > 0:
                logger.warning(f'Feature {column} contains {null_cnt} null values.')
                user_proceed = input('Do you still want to proceed with remapping this feature? (Y/N): ')
                if user_proceed.lower() != 'y':  # If not
                    logger.info(f'Skipping remapping for feature: {column}')  # Log the choice
                    continue  # And move on to the next feature

            # Validate unique values
            unique_vals = sorted(df_final[column].unique())
            if len(unique_vals) < 2:
                logger.warning(f'Feature {column} has fewer than 2 unique values. Skipping remapping.')
                continue

            # Display the feature's current unique values
            print(f'\nCurrent unique values in {column} (alphabetized):')
            for idx, value in enumerate(unique_vals, 1):  # Created enumerated list, starting at index 1
                print(f'{idx}. {value}')

            while True:  # We can justify 'while True' because we have a cancel-out input option
                try:
                    print('\nProvide comma-separated numbers to represent the ordinal order of these values.')
                    print('Example: For [High, Low, Medium], you might enter: 2,0,1')
                    print('Or, for a Likert-type agreement scale [Agree, Disagree, Neither agree nor disagree], '
                          'you might enter: 3,1,2')
                    print('You may also enter "C" to cancel the remapping process for this feature.')

                    user_remap_input = input('\nEnter your mapping values: ')

                    if user_remap_input.lower() == 'c':  # If user cancels
                        logger.info(f'Cancelled remapping for feature: {column}')  # Log the choice
                        break  # And exit the while loop

                    # Convert user remapping input to a list of integers
                    new_vals = [int(x.strip()) for x in user_remap_input.split(',')]

                    # Validate that the user input length matches the number of categories
                    if len(new_vals) != len(unique_vals):
                        raise ValueError('Number of mapping values must match number of categories.')

                    mapping = dict(zip(unique_vals, new_vals))  # Build a mapping dictionary

                    # Display the proposed remapping
                    print('\nProposed mapping:')
                    for old_val, new_val in mapping.items():
                        print(f'- {old_val} → {new_val}')

                    # Ask for user confirmation
                    user_confirm = input('\nDo you want to apply this mapping? (Y/N): ')
                    if user_confirm.lower() == 'y':  # If user confirms
                        df_final[column] = df_final[column].map(mapping)  # Apply the mapping

                        # Add the feature to the list of remapped features
                        remapped_cols.append(column)

                        # Log the remapping
                        logger.info(f'Successfully remapped ordinal feature: {column}')

                        break  # Exit the while loop

                    # Otherwise, restart the remapping attempt
                    else:
                        print('Remapping cancelled. Please try again.')
                        continue

                # Catch input errors
                except ValueError as exc:
                    print(f'Invalid input: {exc}')
                    print('Please try again or enter "C" to cancel.')
                    continue  # Restart the loop

                # Catch other errors
                except Exception as exc:
                    logger.error(f'Error during remapping of feature {column}: {exc}')  # Log the error
                    print('An error occurred. Skipping remapping for this feature.')
                    break  # Exit the loop

        # After all features are processed, log a summary of remapping results
        # If any features were remapped, log the encoded features
        if remapped_cols:
            logger.info('The following ordinal features were remapped to numerical values:')
            for column in remapped_cols:
                logger.info(f'- {column}')

        # If no features were remapped, log that fact
        else:
            logger.info('No ordinal features were remapped.')

    def handle_nums():
        """This internal helper function facilitates the scaling of numerical features, if desired by the user."""
        nonlocal df_final  # Specify that this object comes from the outer scope

        if not num_cols:  # If no columns are tagged as numerical
            logger.info('No numerical features are present in the dataset. Skipping remapping.')  # Log this
            return  # And exit the process

        logger.info(f'The dataset contains {len(num_cols)} numerical features.')  # Log count of numerical features

        # Create list to track which features get scaled for final reporting
        scaled_cols = []

        # Print warning that TADPREPS only supports common scaling methods
        print('\nWARNING: TADPREPS supports only the Standard, Robust, and MinMax scalers.')
        print('For more sophisticated methods (e.g. Quantile or PowerTrans methods), skip this step and write '
              'your own scaler code.')
        # Ask if the user wants to scale any numerical features
        user_scale = input('Do you want to scale any of these features? (Y/N): ')
        if user_scale.lower() != 'y':  # If not
            logger.info('Skipping scaling of numerical features.')  # Log the choice
            return  # And exit the process

        # Ask if user wants a refresher on the three scaling methods
        user_scale_refresh = input('Do you want to see a brief refresher on TADPREPS-supported scalers? (Y/N): ')
        if user_scale_refresh.lower() == 'y':  # If so, display refresher
            print('\nOverview of the Standard, Robust, and MinMax Scalers:'
                  '\nStandard Scaler (Z-score normalization):'
                  '\n- Transforms features to have zero mean and unit variance.'
                  '\n- Best choice for comparing measurements in different units (e.g., combining age, income, '
                  'and test scores).'
                  '\n- Good for methods that assume normally distributed data like linear regression.'
                  '\n- Not ideal when data has many extreme values or outliers.'
                  '\n'
                  '\nRobust Scaler:'
                  '\n- Scales using statistics that are resistant to extreme values.'
                  '\n- Great for data where outliers are meaningful (e.g., rare but important market events).'
                  '\n- Useful for survey data where some respondents give extreme ratings.'
                  '\n- A good choice when you can\'t simply remove outliers because they contain important information.'
                  '\n'
                  '\nMinMax Scaler:'
                  '\n- Scales all values to a fixed range between 0 and 1.'
                  '\n- Good for image pixel data or whenever features must be strictly positive.'
                  '\n- Good for visualization and neural networks that expect bounded inputs.'
                  '\n- Works well with sparse data (data with many zeros).')

        # For each numerical feature
        for column in num_cols:
            print(f'\nProcessing feature: {column}')

            # Ask if user wants to scale this feature
            user_scale_feature = input(f'Do you want to scale {column}? (Y/N): ')
            if user_scale_feature.lower() != 'y':  # If not
                logger.info(f'Skipping scaling for feature: {column}')  # Log that choice
                continue  # And move to the next feature

            # Validate feature before scaling
            try:
                # Check for nulls
                null_cnt = df_final[column].isnull().sum()
                # If nulls are present, log a warning, and ask if user wants to proceed
                if null_cnt > 0:
                    logger.warning(f'Feature {column} contains {null_cnt} null values.')
                    user_proceed = input('Do you still want to proceed with scaling this feature? (Y/N): ')
                    if user_proceed.lower() != 'y':  # If not
                        logger.info(f'Skipping scaling for feature: {column}')  # Log the choice
                        continue  # And move on to the next feature

                # Check for infinite values
                inf_cnt = np.isinf(df_final[column]).sum()
                # If infinite values are present, log a warning, and ask if user wants to proceed
                if inf_cnt > 0:
                    logger.warning(f'Feature {column} contains {inf_cnt} infinite values.')
                    user_proceed = input('Do you want still to proceed with scaling this feature? (Y/N): ')
                    if user_proceed.lower() != 'y':  # If not
                        logger.info(f'Skipping scaling for feature: {column}')  # Log the choice
                        continue  # And move on to the next feature

                # Check for constant/near-constant features, e.g. with only 1 unique value or minimal variance
                if df_final[column].nunique() <= 1:  # If this is so
                    # Log a warning
                    logger.warning(f'Feature {column} has no meaningful variance. Consider dropping this feature.')
                    continue  # And move on to the next feature

                # Check for extreme skewness in the feature
                skewness = df_final[column].skew()  # Calculate skew
                if abs(skewness) > 2:  # Using 2 as threshold for extreme skewness
                    # If skewness is extreme, log a warning and ask if user wants to proceed
                    logger.warning(f'Feature {column} is highly skewed (skewness={skewness:.2f}). '
                                   'Consider transformation before scaling.')
                    user_proceed = input('Do you want still to proceed with scaling this feature? (Y/N): ')
                    if user_proceed.lower() != 'y':  # If not
                        logger.info(f'Skipping scaling for feature: {column}')  # Log the choice
                        continue  # And move on to the next feature

            # Catch validation errors
            except Exception as exc:
                logger.error(f'Error validating feature {column}: {exc}')
                logger.error('Skipping scaling for this feature.')
                continue  # Move to next feature

            # Print (do not log) descriptive statistics for the feature
            print(f'\nDescriptive statistics for {column}:')
            print(df_final[column].describe())

            # Ask if user wants to see a distribution plot for the feature
            user_show_plot = input('\nWould you like to see a histogram of the feature distribution? (Y/N): ')
            if user_show_plot.lower() == 'y':  # If so, send call to Seaborn
                plt.figure(figsize=(12, 8))
                sns.histplot(data=df_final, x=column)
                plt.title(f'Distribution of {column}')
                plt.tight_layout()
                plt.show()

            # Ask user to select a scaling method
            while True:  # We can justify 'while True' because we have a cancel-out input option
                try:
                    print(f'\nSelect scaling method for feature {column}:')
                    print('1. Standard Scaler (Z-score normalization)')
                    print('2. Robust Scaler (uses median and IQR)')
                    print('3. MinMax Scaler (scales to 0-1 range)')
                    print('4. Skip scaling for this feature')
                    scale_method = input('Enter your choice: ')

                    if scale_method == '4':  # If user wants to skip
                        logger.info(f'Skipping scaling for feature: {column}')  # Log that choice
                        break  # Exit the while loop

                    elif scale_method in ['1', '2', '3']:  # If a valid scaling choice is made
                        # Reshape the data for use by scikit-learn
                        reshaped_data = df_final[column].values.reshape(-1, 1)

                        # Instantiate selected scaler and set sacler name
                        if scale_method == '1':
                            scaler = StandardScaler()
                            method_name = 'Standard'
                        elif scale_method == '2':
                            scaler = RobustScaler()
                            method_name = 'Robust'
                        else:  # At this point, scale_method selection must be '3'
                            scaler = MinMaxScaler()
                            method_name = 'MinMax'

                        # Perform scaling and replace original values
                        df_final[column] = scaler.fit_transform(reshaped_data)

                        # Add the feature to the list of scaled features
                        scaled_cols.append(f'{column} ({method_name})')

                        logger.info(f'Applied {method_name} scaling to feature {column}.')  # Log the scaling action
                        break  # Exit the while loop

                    else:  # If an invalid choice was entered
                        print('Invalid input. Please enter 1, 2, 3, or 4.')
                        continue  # Restart the while loop

                # Catch other errors
                except Exception as exc:
                    logger.error(f'Error during scaling of feature {column}: {exc}')  # Log the error
                    print('An error occurred. Skipping scaling for this feature.')
                    break  # Exit the while loop

        # After all features are processed, log a summary of scaling results
        # If any features were scaled, log the scaled features
        if scaled_cols:
            print('-' * 50)  # Visual separator
            logger.info('The following features were scaled:')
            for col in scaled_cols:
                logger.info(f'- {col}')
            print('-' * 50)  # Visual separator

        # If no features were scaled, log that fact
        else:
            print('-' * 50)  # Visual separator
            logger.info('No features were scaled.')
            print('-' * 50)  # Visual separator

    # Call the three helper functions in sequence
    handle_cats()  # Encode categorical features
    handle_ords()  # Transform and remap ordinal features
    handle_nums()  # Scale numerical features

    # Return the final dataset with encoded and scaled features
    return df_final


def export_data(df_final: pd.DataFrame):
    """
    This function handles the exporting of the final, transformed dataset as created by the encode_and_scale()
    function to a static location on disk in one of a few common tabular formats. It is the last step in the TADPREPS
    pipeline.
    Args:
        df_final (pd.DataFrame): The final dataframe created by encode_and_scale().
    Returns:
        This function has no formal return. It instead saves the finished dataframe to disk in accordance with user
        instructions.
    """
    print('Data preparation complete. Preparing for file export.')

    # Define supported export formats and their corresponding file extensions in a dictionary
    supported_formats = {
        '1': ('CSV (.csv)', '.csv'),
        '2': ('Excel (.xlsx)', '.xlsx'),
        '3': ('Pickle (.pkl)', '.pkl'),
        '4': ('Parquet (.parquet)', '.parquet'),
        '5': ('JSON (.json)', '.json')
    }

    # SELECTING EXPORT FORMAT
    while True:  # We can justify 'while True' because we have a cancel-out input option
        try:
            # Display available export formats
            print('\nAvailable export formats:')
            for key, (format_name, _) in supported_formats.items():
                print(f'{key}. {format_name}')
            print('C. Cancel file export')

            # Fetch user export format choice
            format_choice = input('\nSelect an export format (enter format number or "C" to cancel): ').strip()

            # Check for cancellation
            if format_choice.lower() == 'c':  # If user cancels
                logger.info('File export cancelled by user.')  # Log the choice
                return  # And exit the process

            # Validate user format choice
            if format_choice not in supported_formats:
                raise ValueError('Invalid format selection.')

            # Fetch selected format information - name and extension
            format_name, file_extension = supported_formats[format_choice]
            break  # Exit the loop if we get valid input from user

        # Catch input errors
        except ValueError as exc:
            print(f'Invalid input: {exc}')
            continue  # Restart the loop

    # VALIDATING FILENAME
    while True:  # We can justify 'while True' because we need to validate the filename
        try:
            # Fetch desired filename from user (without extension)
            filename = input('\nEnter desired filename (without extension): ').strip()

            # Filename validation
            # Catch empty filename
            if not filename:
                raise ValueError('Filename cannot be empty.')

            # Catch invalid characters
            if any(character in filename for character in r'\/:*?"<>|'):
                raise ValueError('Filename contains invalid characters.')

            # Append appropriate extension
            full_filename = filename + file_extension
            break  # Exit loop if we get valid input from user

        # Catch input errors
        except ValueError as exc:
            print(f'Invalid input: {exc}')
            continue  # Restart the loop

    # VALIDATING FILEPATH
    while True:  # We can justify 'while True' because we need to find a valid directory path for the export
        try:
            # Fetch desired save location from user
            save_path = input('\nEnter the absolute path to the directory where you want to save the file: ').strip()

            # Convert to a Path object and resolve it to an absolute path
            save_dir = Path(save_path).resolve()

            # Create the directory if it doesn't exist
            if not save_dir.exists():
                user_create = input(f'\nDirectory {save_dir} does not exist. Create it? (Y/N): ')
                if user_create.lower() == 'y':
                    try:
                        save_dir.mkdir(parents=True)
                        logger.info(f'Created directory: {save_dir}')

                    # Catch directory creation problems - could be a permissions problem for some users
                    except Exception as exc:
                        raise ValueError(f'Failed to create directory: {exc}')
                else:
                    raise ValueError('Directory does not exist and user declined to create it.')

            # Ensure the path is actually a directory
            if not save_dir.is_dir():
                raise ValueError(f'The path {save_dir} is not a directory.')

            # Activate the relocate_log() function to complete the second logging phase - co-location
            relocate_log(save_dir)

            # Create the full save path using Pathlib - this is preferred b/c it's platform independent
            save_path = save_dir / full_filename

            # Check if the file already exists and give user the opportunity to overwrite it
            if save_path.exists():
                user_overwrite = input(f'\nFile {full_filename} already exists. Overwrite? (Y/N): ')
                if user_overwrite.lower() != 'y':
                    raise ValueError('User declined to overwrite existing file.')

            break  # Exit loop if we get valid input from user

        # Catch input errors
        except ValueError as exc:
            print(f'Invalid input: {exc}')
            continue  # Restart the loop

        # Catch other errors
        except Exception as exc:
            print(f'Error processing directory path: {exc}')
            continue  # Restart the loop

    # EXPORTING FILE
    # Size validation before export
    try:
        estimated_size_mb = df_final.memory_usage(deep=True).sum() / (1024 * 1024)
        if estimated_size_mb > 1000:  # If over 1GB
            logger.warning(f'Large file size detected ({estimated_size_mb:.2f} MB). Export may take some time.')

    # Catch file size/memory usage estimation errors
    except Exception as exc:
        logger.warning(f'Unable to estimate file size: {exc}')

    # Exporting file
    try:
        print('Attempting file export...')
        # Export the file based on chosen format
        if format_choice == '1':  # CSV
            df_final.to_csv(save_path, index=False)

        elif format_choice == '2':  # Excel
            df_final.to_excel(save_path, index=False)

        elif format_choice == '3':  # Pickle
            df_final.to_pickle(save_path)

        elif format_choice == '4':  # Parquet
            df_final.to_parquet(save_path, index=False)

        else:  # JSON
            df_final.to_json(save_path, orient='records')

        # Log successful file export
        logger.info(f'Successfully exported prepared dataset as {format_name}')
        logger.info(f'Filename: {full_filename}')
        logger.info(f'File saved to: {save_path}')
        logger.info('TADPREPS execution is complete.')

    # Catch all other errors and log
    except Exception as exc:
        logger.error(f'Error during file export: {exc}')
        print('Failed to export file. Please check the logs for details.')


def main():
    """
    This is the main function for the TADPREPS (Tabular Automated Data PREParation System) program.
    It orchestrates the data preparation workflow by calling the core functions in the proper sequence
    and managing the dataframe transformations throughout the pipeline.

    The workflow consists of:
    1. Loading the data file
    2. Printing basic file information for the user
    3. Trimming the dataset (handling missing values and sub-setting the data, if desired)
    4. Renaming features and adding suffixes
    5. Printing feature statistics
    6. Imputing missing values
    7. Encoding and scaling features
    8. Exporting the prepared dataset

    Returns:
        None. This function handles the entire TADPREPS workflow and ends with the file export process.
    """
    try:
        # Print welcome message and basic instructions
        print('Welcome to TADPREPS (Tabular Automated Data PREParation System)')
        print('This program will guide you through preparing your tabular dataset for further analysis.')
        print('Follow the prompts and make selections when asked to do so.')

        # Stage 1: Load the data file
        print('\nSTAGE 1: LOADING DATA FILE')
        print('-' * 50)
        df_full = load_file()

        # Stage 2: Print basic file information
        print('\nSTAGE 2: DISPLAYING FILE INFORMATION')
        print('-' * 50)
        print_file_info(df_full)

        # Stage 3: Allow user to trim the dataset if desired
        print('\nSTAGE 3: TRIMMING DATASET')
        print('-' * 50)
        df_trimmed = trim_file(df_full)

        # Stage 4: Allow user to rename features and add feature-type suffixes
        print('\nSTAGE 4: RENAMING FEATURES')
        print('-' * 50)
        df_renamed = rename_features(df_trimmed)

        # Stage 5: Print feature statistics and get feature type lists
        print('\nSTAGE 5: ANALYZING FEATURES')
        print('-' * 50)
        cat_cols, ord_cols, num_cols = print_feature_stats(df_renamed)

        # Stage 6: Handle missing value imputation
        print('\nSTAGE 6: HANDLING/IMPUTING MISSING VALUES')
        print('-' * 50)
        df_imputed = impute_missing_data(df_renamed)

        # Stage 7: Perform encoding and scaling
        print('\nSTAGE 7: ENCODING AND SCALING FEATURES')
        print('-' * 50)
        df_final = encode_and_scale(df_imputed, cat_cols, ord_cols, num_cols)

        # Stage 8: Export the prepared dataset
        print('\nSTAGE 8: EXPORTING FINISHED DATASET')
        print('-' * 50)
        export_data(df_final)

    # Catch process exceptions and log
    except Exception as exc:
        logger.error(f'An unexpected error occurred in the main TADPREPS workflow: {exc}')
        logger.error('TADPREPS execution terminated due to an error. See log for details.')
        sys.exit(1)  # Exit program


if __name__ == '__main__':
    main()
