"""
This is the TADPREPS codebase. All core functions will be defined internally (so no additional files are necessary at
runtime) and the logging file will be created in the same working directory as the script itself.
"""

# Library imports
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import logging
import sys

# Set up error logging
# We persist with "%-type" formatting to preserve backward compatibility
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('tadpreps_runtime_log.log'), logging.StreamHandler(sys.stdout)]
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
