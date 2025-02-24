import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


def _df_info_core(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Core function to print general top-level information about the full, unaltered datafile for the user.

    Args:
        df (pd.DataFrame): A Pandas dataframe containing the full, unaltered dataset.
        verbose (bool): Whether to print more detailed information about the file. Defaults to True.

    Returns:
        None. This is a void function.
    """
    # Print number of instances in file
    print(f'The dataset has {df.shape[0]} instances.')  # [0] is rows

    # Print number of features in file
    print(f'The dataset has {df.shape[1]} features.')  # [1] is columns

    # Handle instances with missing values
    row_nan_cnt = df.isnull().any(axis=1).sum()  # Compute count of instances with any missing values
    row_nan_rate = (row_nan_cnt / len(df) * 100).round(2)  # Compute rate
    print(f'{row_nan_cnt} instance(s) ({row_nan_rate}%) contain at least one missing value.')

    # Handle duplicate instances
    # NOTE: This check runs regardless of verbose status because it can help catch a data integrity error
    row_dup_cnt = df.duplicated().sum()  # Compute count of duplicate instances
    if row_dup_cnt > 0:  # If any such instances exist
        row_dup_rate = ((row_dup_cnt / len(df)) * 100)  # Compute duplication rate
        print(f'\nThe dataset contains {row_dup_cnt} duplicate instance(s). ({row_dup_rate:.2f}% of all instances)')
        if verbose:
            print('WARNING: If duplicate instances should not exist in your data, this may indicate an error in how '
                  'your data were collected or imported.')

    # Subsequent data checks only run if verbose is True
    if verbose:
        # Create a list of near-constant features, if any exist
        near_constant_cols = [(column,
                               value_counts.index[0],  # Index 0 is the most frequent value in the column
                               value_counts.iloc[0] * 100)
                              # For each column
                              for column in df.columns
                              # Check if most frequent value in column exceeds 95% of data present
                              if (value_counts := df[column].value_counts(normalize=True)).iloc[0] > 0.95]  # Walrus!

        if near_constant_cols:
            print(f'\nALERT: {len(near_constant_cols)} feature(s) has very low variance (>95% single-value):')
            for column, value, rate in near_constant_cols:
                print(f'- {column} (Dominant value: {value}, Dominance rate: {rate:.2f}%)')

        # Create a list of features containing infinite values, if any exist
        inf_cols = [(column, np.isinf(df[column]).sum())  # Column name and count of inf values
                    for column in df.select_dtypes(include=['int64', 'float64']).columns  # For each numeric column
                    if np.isinf(df[column]).sum() > 0]  # Append if the column contains any inf values

        if inf_cols:
            print(f'\nALERT: {len(inf_cols)} feature(s) contains infinite values:')
            for column, inf_count in inf_cols:
                print(f'- {column} contains {inf_count} infinite value(s)')

        # Create a list of features containing empty strings (i.e. distinct from NULL/NaN), if any exist
        empty_string_cols = [(column, (df[column] == '').sum())  # Column name and count of empty strings
                             for column in df.select_dtypes(include=['object']).columns  # For each string column
                             if (df[column] == '').sum() > 0]  # If column contains any empty strings

        if empty_string_cols:
            print(f'\nALERT: {len(empty_string_cols)} feature(s) contains non-NULL empty strings '
                  f'(e.g. values of "", distinct from NULL/NaN):')
            for column, empty_count in empty_string_cols:
                print(f'- {column} contains {empty_count} non-NULL empty string(s)')

        # Finally, print names and datatypes of features in file
        print('\nNAMES AND DATATYPES OF FEATURES:')
        print('-' * 50)  # Visual separator
        print(df.info(verbose=True, memory_usage=True, show_counts=True))
        print('-' * 50)  # Visual separator


def _reshape_core(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Core function for reshaping a DataFrame by handling missing values, dropping features, and subsetting data.

    Args:
        df (pd.DataFrame): Input DataFrame to reshape
        verbose (bool): Whether to print detailed information about operations. Defaults to True.

    Returns:
        pd.DataFrame: Reshaped DataFrame

    Raises:
        ValueError: If invalid indices are provided for column dropping
        ValueError: If an invalid subsetting proportion is provided
    """
    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning data reshape process.')
        print('-' * 50)  # Visual separator

    row_missing_cnt = df.isnull().any(axis=1).sum()  # Compute count
    # Ask if the user wants to delete *all* instances with any missing values, if any exist
    if row_missing_cnt > 0:
        user_drop_na = input('Do you want to drop all instances with *any* missing values? (Y/N): ')
        if user_drop_na.lower() == 'y':
            df = df.dropna()
            if verbose:
                print(f'After deletion of {row_missing_cnt} instances with missing values, {len(df)} instances remain.')

    # Ask if the user wants to drop any of the columns/features in the dataset
    user_drop_cols = input('\nDo you want to drop any of the features in the dataset? (Y/N): ')
    if user_drop_cols.lower() == 'y':
        print('The full set of features in the dataset is:')
        for col_idx, column in enumerate(df.columns, 1):  # Create enumerated list of features starting at 1
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                drop_cols_input = input('\nEnter the index integers of the features you wish to drop '
                                        '(comma-separated) or enter "C" to cancel: ')

                # Check for user cancellation
                if drop_cols_input.lower() == 'c':
                    if verbose:
                        print('Feature deletion cancelled.')
                    break

                # Create list of column indices to drop
                drop_cols_idx = [int(idx.strip()) for idx in drop_cols_input.split(',')]  # Splitting on comma

                # Verify that all index numbers of columns to be dropped are valid/in range
                if not all(1 <= idx <= len(df.columns) for idx in drop_cols_idx):  # Using a generator
                    raise ValueError('Some feature index integers entered are out of range/invalid.')

                # Convert specified column numbers to actual column names
                drop_cols_names = [df.columns[idx - 1] for idx in drop_cols_idx]  # Subtracting 1 from indices

                # Drop the columns
                df = df.drop(columns=drop_cols_names)
                if verbose:
                    print('-' * 50)  # Visual separator
                    print(f'Dropped features: {",".join(drop_cols_names)}')  # Note dropped columns
                    print('-' * 50)  # Visual separator
                break

            # Catch invalid user input
            except ValueError:
                print('Invalid input. Please enter valid feature index integers separated by commas.')
                continue  # Restart the loop

    # Ask if the user wants to sub-set the data
    user_subset = input('Do you want to sub-set the data by randomly deleting a specified proportion of '
                        'instances? (Y/N): ')
    if user_subset.lower() == 'y':
        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or '
                                     'enter "C" to cancel: ')

                # Check for user cancellation
                if subset_input.lower() == 'c':
                    if verbose:
                        print('Random sub-setting cancelled.')
                    break

                subset_rate = float(subset_input)  # Convert string input to float
                if 0 < subset_rate < 1:  # If the float is valid (i.e. between 0 and 1)
                    retain_rate = 1 - subset_rate  # Compute retention rate
                    retain_row_cnt = int(len(df) * retain_rate)  # Select count of rows to keep in subset

                    df = df.sample(n=retain_row_cnt)  # No random state set b/c we want true randomness
                    if verbose:
                        print(f'Randomly dropped {subset_rate}% of instances. {retain_row_cnt} instances remain.')
                    break

                # Catch user input error for invalid/out-of-range float
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Catch outer-level user input errors
            except ValueError:
                print('Invalid input. Enter a float value between 0.0 and 1.0 or enter "C" to cancel.')
                continue  # Restart the loop
    if verbose:
        print('-' * 50)  # Visual separator
        print('Data reshape complete. Returning modified dataframe.')
        print('-' * 50)  # Visual separator

    return df  # Return the trimmed dataframe


def _subset_core(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Core function for subsetting a DataFrame via random sampling (with or without seed), stratified sampling,
    or time-based instance selection for timeseries data.

    Args:
        df (pd.DataFrame): Input DataFrame to subset
        verbose (bool): Whether to print detailed information about operations. Defaults to True.

    Returns:
        pd.DataFrame: Subsetted DataFrame

    Raises:
        ValueError: If invalid subsetting proportion is provided
        ValueError: If invalid date range is provided for timeseries subsetting
    """
    if verbose:
        print('-' * 50)
        print('Beginning data subset process.')
        print('-' * 50)

    # Build list of categorical columns, if any exist
    cat_cols = [column for column in df.columns
                if pd.api.types.is_object_dtype(df[column]) or
                isinstance(df[column].dtype, type(pd.Categorical.dtype))]

    # Check if data might be timeseries in form
    datetime_cols = []

    for column in df.columns:
        # Check if column is already datetime type
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_cols.append(column)
        # For string columns, try to parse as datetime
        elif pd.api.types.is_object_dtype(df[column]):
            try:
                # Try to parse first non-null value
                first_valid = df[column].dropna().iloc[0]
                pd.to_datetime(first_valid)
                datetime_cols.append(column)

            # If the process doesn't work, move to the next column
            except (ValueError, IndexError):
                continue

    is_datetime_index = pd.api.types.is_datetime64_any_dtype(df.index)
    has_datetime = bool(datetime_cols or is_datetime_index)

    # Convert any detected datetime string columns to datetime type
    if datetime_cols:
        for column in datetime_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                try:
                    df[column] = pd.to_datetime(df[column])

                # If conversion fails, remove that feature from the list of datetime features
                except ValueError:
                    datetime_cols.remove(column)

    if verbose:
        print('NOTE - TADPREP supports:'
              '\n - Seeded and unseeded random sampling'
              '\n - Stratified random sampling (if categorical features are present)'
              '\n - Date-based instance deletion (if the data are a timeseries).')

    # Present subset options based on data characteristics
    print('\nAvailable subset methods for this dataset:')
    print('1. Unseeded Random sampling (For true randomness)')
    print('2. Seeded Random sampling (For reproducibility)')

    # Only offer stratified sampling if categorical columns are present
    if cat_cols:
        print('3. Stratified random sampling (Preserves ratio of categories in data)')

    # Only offer time-based subsetting if datetime elements exist
    if has_datetime:
        print('4. Time-based boundaries for subset (Select only data within a specified time range)')

    while True:
        try:
            method = input('\nSelect a subset method or enter "C" to cancel: ').strip()

            # Handle cancellation
            if method.lower() == 'c':
                if verbose:
                    print('Subsetting cancelled. Input dataframe was not modified.')
                return df

            # Validate input based on available options
            valid_options = ['1', '2']

            # Show option for stratified sampling only if categorical features exist
            if cat_cols:
                valid_options.append('3')

            # Show option for time-based deletion only if data are a timeseries
            if has_datetime:
                valid_options.append('4')

            # Handle invalid user input
            if method not in valid_options:
                raise ValueError('Invalid selection.')
            break

        # Handle non-numerical user input
        except ValueError:
            print('Invalid input. Please select from the options provided.')
            continue

    # Handle true random sampling (unseeded)
    if method == '1':
        if verbose:
            print('-' * 50)  # Visual separator
            print('Proceeding with true (unseeded) random sampling...')
            print('-' * 50)  # Visual separator

        while True:
            try:
                # Fetch subsetting proportion
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or enter "C" to cancel: ')

                # Handle user cancellation
                if subset_input.lower() == 'c':
                    if verbose:
                        print('Random subsetting cancelled. Input dataframe was not modified.')
                    return df

                subset_rate = float(subset_input)
                if 0 < subset_rate < 1:
                    retain_rate = 1 - subset_rate  # Calculate retention rate
                    retain_row_cnt = int(len(df) * retain_rate)  # Construct integer count for instance deletion
                    df = df.sample(n=retain_row_cnt)  # Perform random sampling

                    if verbose:
                        print(f'Randomly dropped {subset_rate:.1%} of instances. {retain_row_cnt} instances remain.')
                    break

                # Handle invalid user input
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Handle non-numerical user input
            except ValueError:
                print('Invalid input. Enter a float value between 0.0 and 1.0 or enter "C" to cancel.')
                continue

    # Handle random sampling (seeded)
    elif method == '2':
        if verbose:
            print('-' * 50)  # Visual separator
            print('Proceeding with reproducible (seeded) random sampling...')
            print('-' * 50)  # Visual separator

        while True:
            try:
                # Fetch subsetting proportion
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or enter "C" to cancel: ')

                # Handle user cancellation
                if subset_input.lower() == 'c':
                    if verbose:
                        print('Random subsetting cancelled. Input dataframe was not modified.')
                    return df

                subset_rate = float(subset_input)
                if 0 < subset_rate < 1:
                    # Use a fixed seed for reproducibility
                    seed = 4  # Because 4 is my lucky number and using 42 is cringe
                    retain_rate = 1 - subset_rate  # Calculate retention rate
                    retain_row_cnt = int(len(df) * retain_rate)  # Construct integer count for instance deletion
                    df = df.sample(n=retain_row_cnt, random_state=seed)  # Perform seeded random sampling

                    if verbose:
                        print(f'Randomly dropped {subset_rate:.1%} of instances using reproducible sampling.')
                        print(f'{retain_row_cnt} instances remain.')
                    break

                # Handle invalid user input
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Handle non-numerical user input
            except ValueError:
                print('Invalid input. Enter a float value between 0.0 and 1.0 or enter "C" to cancel.')
                continue

    # Handle stratified sampling
    elif method == '3':
        if verbose:
            print('-' * 50)  # Visual separator
            print('Proceeding with stratified random sampling...')
            print('-' * 50)  # Visual separator

            user_explain = input('Would you like to see a brief explanation of stratified sampling? (Y/N): ')
            if user_explain.lower() == 'y':
                print('-' * 50)  # Visual separator
                print('Overview of Stratified Random Sampling:'
                      '\n- This method samples data while maintaining proportions in a specific feature.'
                      '\n- You will select a single categorical feature to "stratify by."'
                      '\n- The proportions in that feature (and *only* in that feature) will be preserved in your '
                      'sampled data.'
                      '\n- Example: If you stratify by gender in a customer dataset that is 80% male and 20% female,'
                      '\n  your sample will maintain this 80/20 split, even if other proportions in the data change.'
                      '\n- This process prevents underrepresentation of minority categories in your chosen feature.'
                      '\n- This method is most useful when you have important categories that are imbalanced.')
                print('-' * 50)  # Visual separator

        # Display available categorical features
        print('\nAvailable categorical features:')
        for idx, col in enumerate(cat_cols, 1):
            print(f'{idx}. {col}')

        while True:
            try:
                strat_input = input('\nEnter the number of the feature to stratify by: ')
                strat_idx = int(strat_input) - 1

                # Handle bad user input
                if not 0 <= strat_idx < len(cat_cols):
                    raise ValueError('Invalid feature number.')

                strat_col = cat_cols[strat_idx]  # Set column to stratify by

                # Fetch subsetting proportion
                subset_input = input('Enter the proportion of instances to DROP (0.0-1.0) or "C" to cancel: ')

                # Handle user cancellation
                if subset_input.lower() == 'c':
                    if verbose:
                        print('Stratified subsetting cancelled. Input dataframe was not modified.')
                    return df

                subset_rate = float(subset_input)
                if 0 < subset_rate < 1:
                    retain_rate = 1 - subset_rate  # Calculate retention rate
                    start_size = len(df)  # Store original dataset size

                    # Perform stratified sampling with minimum instance guarantee
                    stratified_dfs = []
                    unique_values = df[strat_col].unique()

                    # Check if we can maintain at least one instance per category
                    min_instances_needed = len(unique_values)
                    total_retain = int(len(df) * retain_rate)

                    if total_retain < min_instances_needed:
                        print(f'\nWARNING: Cannot maintain stratification with {retain_rate:.1%} retention rate.')
                        print(f'Need at least {min_instances_needed} instances to maintain one per category.')
                        user_proceed = input('Would you like to retain one instance per category instead? (Y/N): ')
                        if user_proceed.lower() != 'y':
                            continue
                        # Adjust to keep one per category
                        retain_rate = min_instances_needed / len(df)
                        print(f'\nAdjusted retention rate to {retain_rate:.1%} to maintain stratification.')

                    for value in unique_values:
                        subset = df[df[strat_col] == value]
                        retain_count = max(1, int(len(subset) * retain_rate))  # Ensure at least 1 instance
                        sampled = subset.sample(n=retain_count)
                        stratified_dfs.append(sampled)

                    df = pd.concat(stratified_dfs)

                    # Calculate actual retention rate achieved
                    true_retain_rate = len(df) / start_size
                    true_drop_rate = 1 - true_retain_rate

                    if abs(subset_rate - true_drop_rate) > 0.01:  # If difference is more than 1%
                        print(f'\nNOTE: To maintain proportional stratification, the actual drop rate was '
                              f'adjusted to {true_drop_rate:.1%}.')
                        print(f'(You requested: {subset_rate:.1%})')

                    if verbose:
                        print(f'\nPerformed stratified sampling by "{strat_col}".')
                        print(f'{len(df)} instances remain.')
                    break

                # Handle invalid user input
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Handle all other input problems
            except ValueError as exc:
                print(f'Invalid input: {str(exc)}')
                continue

    # Handle time-based subsetting
    elif method == '4':
        if verbose:
            print('-' * 50)  # Visual separator
            print('Proceeding with time-based subsetting...')
            print('-' * 50)  # Visual separator

        # Identify datetime column
        if is_datetime_index:
            time_col = df.index
            col_name = 'index'

        # If there's more than one datetime feature, allow user to select which one to use
        else:
            if len(datetime_cols) > 1:
                print('\nAvailable datetime features:')
                for idx, col in enumerate(datetime_cols, 1):
                    print(f'{idx}. {col}')

                while True:
                    try:
                        col_input = input('\nSelect the feature you want to use to create your subset "boundaries": ')
                        col_idx = int(col_input) - 1
                        if not 0 <= col_idx < len(datetime_cols):
                            raise ValueError('Invalid feature number.')
                        col_name = datetime_cols[col_idx]
                        time_col = df[col_name]
                        break

                    # Handle bad user input
                    except ValueError:
                        print('Invalid input. Please enter a valid number.')
            else:
                col_name = datetime_cols[0]
                time_col = df[col_name]

        # Determine time frequency from data
        if isinstance(time_col, pd.Series):
            time_diffs = time_col.sort_values().diff().dropna().unique()
        else:
            time_diffs = pd.Series(time_col).sort_values().diff().dropna().unique()

        # Fetch most common temporal difference between instances to identify the timeseries frequency
        if len(time_diffs) > 0:
            most_common_diff = pd.Series(time_diffs).value_counts().index[0]
            freq_hours = most_common_diff.total_seconds() / 3600  # Use hours as a baseline computation

            # Define frequency type/level using most common datetime typologies
            if freq_hours < 1:
                freq_str = 'sub-hourly'
            elif freq_hours == 1:
                freq_str = 'hourly'
            elif freq_hours == 24:
                freq_str = 'daily'
            elif 24 * 28 <= freq_hours <= 24 * 31:
                freq_str = 'monthly'
            elif 24 * 89 <= freq_hours <= 24 * 92:
                freq_str = 'quarterly'
            elif 24 * 365 <= freq_hours <= 24 * 366:
                freq_str = 'yearly'

            # In case the data are just weird
            else:
                freq_str = 'irregular'

        # In case we literally couldn't compute the time deltas
        else:
            freq_str = 'undetermined'

        # Fetch data range and some example timestamps
        min_date = time_col.min()
        max_date = time_col.max()
        example_stamps = sorted(time_col.sample(n=min(3, len(time_col))).dt.strftime('%Y-%m-%d %H:%M:%S'))

        if verbose:
            print(f'\nData frequency/time aggregation level appears to be {freq_str}.')
        print(f'Your timeseries spans from {min_date} to {max_date}.')
        if verbose:
            print('\nExample timestamps from your data:')
            for stamp in example_stamps:
                print(f'- {stamp}')
            print('\nEnter time boundaries in any standard format (YYYY-MM-DD, MM/DD/YYYY, etc.) or press Enter to use '
                  'the earliest/latest timestamp')

        while True:
            try:
                # Fetch start boundary
                start_input = input('\nEnter the earliest/starting time boundary for the subset, or press Enter '
                                    'to use the earliest timestamp: ').strip()
                if start_input:
                    start_date = pd.to_datetime(start_input)
                else:
                    start_date = min_date

                # Fetch end boundary
                end_input = input('\nEnter the latest/ending time boundary for the subset, or press Enter to use '
                                  'the latest timestamp: ').strip()
                if end_input:
                    end_date = pd.to_datetime(end_input)
                else:
                    end_date = max_date

                # Validate the user's date range
                if start_date > end_date:
                    raise ValueError('Start time must be before end time.')

                # Apply the time filter
                if is_datetime_index:
                    df_filtered = df[start_date:end_date]
                else:
                    df_filtered = df[(df[col_name] >= start_date) & (df[col_name] <= end_date)]

                # Check if any data remains after filtering
                if df_filtered.empty:
                    print(f'WARNING: No instances found between {start_date} and {end_date}.')
                    print('Please provide a different date range.')
                    continue  # Go back to date input

                df = df_filtered  # Only update the df if we actually found data in the range defined by the user

                if verbose:
                    print(f'\nRetained instances from {start_date} to {end_date}.')
                    print(f'{len(df)} instances remain after subsetting.')
                break

            except ValueError as exc:
                print(f'Invalid input: {str(exc)}')
                continue

    if verbose:
        print('-' * 50)  # Visual separator
        print('Data subsetting complete. Returning modified dataframe.')
        print('-' * 50)  # Visual separator

    return df  # Return modified dataframe


def _rename_and_tag_core(df: pd.DataFrame, verbose: bool = True, tag_features: bool = False) -> pd.DataFrame:
    """
    Core function to rename features and to append the '_ord' and/or '_target' suffixes to ordinal or target features,
    if desired by the user.

    Args:
        df (pd.DataFrame): Input DataFrame to reshape.
        verbose (bool): Whether to print detailed information about operations. Defaults to True.
        tag_features (bool): Whether to activate the feature-tagging process. Defaults to False.

    Returns:
        pd.DataFrame: Reshaped DataFrame

    Raises:
        ValueError: If invalid indices are provided for column renaming
        ValueError: If any other invalid input is provided
    """
    def valid_identifier_check(name: str) -> tuple[bool, str]:
        """Helper function to validate whether a proposed feature name follows Python naming conventions."""
        # Check using valid identifier method
        if not name.isidentifier():
            return False, ('New feature name must be a valid Python identifier (i.e. must contain only letters, '
                           'numbers, and underscores, and cannot start with a number)')
        else:
            return True, ''

    def problem_chars_check(name: str) -> list[str]:
        """Helper function to identify potentially problematic characters in a proposed feature name."""
        # Instantiate empty list to hold warnings
        char_warnings = []
        # Space check
        if ' ' in name:
            char_warnings.append('Contains spaces')

        # Special character check
        if any(char in name for char in '!@#$%^&*()+=[]{}|\\;:"\',.<>?/'):
            char_warnings.append('Contains special characters')

        # Double underscore check
        if '__' in name:
            char_warnings.append('Contains double underscores')

        # Return warning list
        return char_warnings

    def antipattern_check(name: str) -> list[str]:
        """Helper function to check for common naming anti-patterns in a proposed feature name."""
        # Instantiate empty list to hold warnings
        pattern_warnings = []

        # Check for common problematic anti-patterns
        if name[0].isdigit():
            pattern_warnings.append('Name starts with a number')
        if name.isupper():
            pattern_warnings.append('Name is all uppercase')
        if len(name) <= 2:
            pattern_warnings.append('Name is very short')
        if len(name) > 30:
            pattern_warnings.append('Name is very long')

        # Return warning list
        return pattern_warnings

    def python_keyword_check(name: str) -> bool:
        """Helper function to check if a proposed feature name conflicts with Python keywords."""
        import keyword  # We can do this in one step with the keyword library
        return keyword.iskeyword(name)

    # Track all rename operations for end-of-process summary
    rename_tracker = []

    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning feature renaming process.')
        print('-' * 50)  # Visual separator

    while True:  # We can justify 'while True' because we have a cancel-out input option
        try:
            if verbose:
                print('The list of features currently present in the dataset is:')
            else:
                print('Features:')

            for col_idx, column in enumerate(df.columns, 1):  # Create enumerated list of features starting at 1
                print(f'{col_idx}. {column}')

            rename_cols_input = input('\nEnter the index integer of the feature you want to rename, enter "S" to skip '
                                      'this step, or enter "E" to exit the renaming/tagging process: ')

            # Check for user skip
            if rename_cols_input.lower() == 's':
                if verbose:
                    print('Feature renaming skipped.')
                break

            # Check for user process exit
            elif rename_cols_input.lower() == 'e':
                if verbose:
                    print('Exiting process. Dataframe was not modified.')
                return df

            col_idx = int(rename_cols_input)  # Convert input to integer
            if not 1 <= col_idx <= len(df.columns):  # Validate entry
                raise ValueError('Column index is out of range.')

            # Get new column name from user
            col_name_old = df.columns[col_idx - 1]
            col_name_new = input(f'Enter new name for feature "{col_name_old}": ').strip()

            # Run Python identifier validation
            valid_status, valid_msg = valid_identifier_check(col_name_new)
            if not valid_status:
                print(f'ERROR: Invalid feature name: {valid_msg}')
                continue

            # Check if proposed name is a Python keyword
            if python_keyword_check(col_name_new):
                print(f'"{col_name_new}" is a Python keyword and cannot be used as a feature name. '
                      f'Please choose a different name.')
                continue

            # Check for problematic characters
            char_warnings = problem_chars_check(col_name_new)
            if char_warnings and verbose:
                print('\nWARNING: Potential character-choice issues with proposed feature name:')
                for warning in char_warnings:
                    print(f'- {warning}')
                if input('Do you want to use your proposed feature name anyway? (Y/N): ').lower() != 'y':
                    continue

            # Check for naming anti-patterns
            pattern_warnings = antipattern_check(col_name_new)
            if pattern_warnings and verbose:
                print('\nWARNING: Proposed feature name does not follow best practices for anti-pattern avoidance:')
                for warning in pattern_warnings:
                    print(f'- {warning}')
                if input('Do you want to use your proposed feature name anyway? (Y/N): ').lower() != 'y':
                    continue

            # Validate name to make sure it doesn't already exist
            if col_name_new in df.columns:
                print(f'Feature name "{col_name_new}" already exists. Choose a different name.')
                continue

            # Preview and approve each change in verbose mode only
            if verbose:
                print(f'\nProposed feature name change: "{col_name_old}" -> "{col_name_new}"')
                if input('Apply this change? (Y/N): ').lower() != 'y':
                    continue

            # Rename column in-place
            df = df.rename(columns={col_name_old: col_name_new})

            # Track the rename operation
            rename_tracker.append({'old_name': col_name_old, 'new_name': col_name_new, 'type': 'rename'})

            # Print renaming summary message in verbose mode only
            if verbose:
                print('-' * 50)  # Visual separator
                print(f'Successfully renamed feature "{col_name_old}" to "{col_name_new}".')
                print('-' * 50)  # Visual separator

            # Ask if user wants to rename another column
            if input('Do you want to rename another feature? (Y/N): ').lower() != 'y':
                break

        # Catch input errors
        except ValueError as exc:
            print(f'Invalid input: {exc}')

    if tag_features:
        if verbose:
            print('-' * 50)  # Visual separator
            print('Beginning ordinal feature tagging process.')
            print('-' * 50)  # Visual separator
            print('You may now select any ordinal features which you know to be present in the dataset and append the '
                  '"_ord" suffix to their feature names.')
            print('If no ordinal features are present in the dataset, enter "S" to skip this step.')

        else:
            print('\nOrdinal feature tagging:')

        print('\nFeatures:')
        for col_idx, column in enumerate(df.columns, 1):
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                ord_input = input('\nEnter the index integers of ordinal features (comma-separated), enter "S" to skip '
                                  'this step, or enter "E" to exit the renaming process: ')

                # Check for user skip
                if ord_input.lower() == 's':
                    if verbose:
                        print('Ordinal feature tagging skipped.')  # Note the cancellation
                    break  # Exit the loop

                # Check for user process exit
                elif ord_input.lower() == 'e':
                    if verbose:
                        print('Exiting process. Dataframe was not modified.')
                    return df

                ord_idx_list = [int(idx.strip()) for idx in ord_input.split(',')]  # Create list of index integers

                # Validate that all entered index integers are in range
                if not all(1 <= idx <= len(df.columns) for idx in ord_idx_list):
                    raise ValueError('Some feature indices are out of range.')

                ord_names_pretag = [df.columns[idx - 1] for idx in ord_idx_list]  # Create list of pretag names

                # Generate mapper for renaming columns with '_ord' suffix
                ord_rename_map = {name: f'{name}_ord' for name in ord_names_pretag if not name.endswith('_ord')}

                # Validate that tags for the selected columns are not already present (i.e. done pre-import)
                if not ord_rename_map:  # If the mapper is empty
                    print('WARNING: All selected features are already tagged as ordinal.')  # Warn the user
                    print('Skipping ordinal tagging.')
                    break

                df.rename(columns=ord_rename_map, inplace=True)  # Perform tagging
                # Track ordinal tagging operations
                for old_name, new_name in ord_rename_map.items():
                    rename_tracker.append({'old_name': old_name, 'new_name': new_name, 'type': 'ordinal_tag'})

                # Print ordinal tagging summary message in verbose mode only
                if verbose:
                    print('-' * 50)  # Visual separator
                    print(f'Tagged the following features as ordinal: {", ".join(ord_rename_map.keys())}')
                break

            # Catch invalid input
            except ValueError as exc:
                print(f'Invalid input: {exc}')
                continue

        if verbose:
            print('-' * 50)  # Visual separator
            print('Beginning target feature tagging process.')
            print('-' * 50)  # Visual separator
            print('You may now select any target features which you know to be present in the dataset and append the '
                  '"_target" suffix to their feature names.')
            print('If no target features are present in the dataset, enter "S" to skip this step.')

        else:
            print('\nTarget feature tagging:')

        print('\nFeatures:')
        for col_idx, column in enumerate(df.columns, 1):
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                target_input = input('\nEnter the index integers of target features (comma-separated), enter "S" to '
                                     'skip this step, or enter "E" to exit the renaming process: ')

                # Check for user cancellation
                if target_input.lower() == 's':
                    if verbose:
                        print('Target feature tagging skipped.')
                    break

                # Check for user process exit
                elif target_input.lower() == 'e':
                    if verbose:
                        print('Exiting process. Dataframe was not modified.')
                    return df

                target_idx_list = [int(idx.strip()) for idx in target_input.split(',')]  # Create list of index integers

                # Validate that all entered index integers are in range
                if not all(1 <= idx <= len(df.columns) for idx in target_idx_list):
                    raise ValueError('Some feature indices are out of range.')

                target_names_pretag = [df.columns[idx - 1] for idx in target_idx_list]  # List of pretag names

                # Generate mapper for renaming columns with '_target' suffix
                target_rename_map = {name: f'{name}_target' for name in target_names_pretag if
                                     not name.endswith('_target')}

                # Validate that tags for the selected columns are not already present (i.e. done pre-import)
                if not target_rename_map:  # If the mapper is empty
                    print('WARNING: All selected features are already tagged as targets.')  # Warn the user
                    print('Skipping target tagging.')
                    break

                df = df.rename(columns=target_rename_map)  # Perform tagging
                # Track target tagging operations
                for old_name, new_name in target_rename_map.items():
                    rename_tracker.append({'old_name': old_name, 'new_name': new_name, 'type': 'target_tag'})

                # Print ordinal tagging summary message in verbose mode only
                if verbose:
                    print('-' * 50)  # Visual separator
                    print(f'Tagged the following features as targets: {", ".join(target_rename_map.keys())}')
                    print('-' * 50)  # Visual separator
                break

            # Catch invalid input
            except ValueError as exc:
                print(f'Invalid input: {exc}')
                continue  # Restart the loop

    if verbose and rename_tracker:  # Only show summary if verbose mode is active and changes were made
        print('\nSUMMARY OF CHANGES:')
        print('-' * 50)

        # Create summary DataFrame
        summary_df = pd.DataFrame(rename_tracker)

        # Group by operation type and display changes
        for op_type in summary_df['type'].unique():
            op_changes = summary_df[summary_df['type'] == op_type]
            if op_type == 'rename':
                print('Features Renamed:')
            elif op_type == 'ordinal_tag':
                print('\nOrdinal Tags Added:')
            else:  # target_tag
                print('\nTarget Tags Added:')

            # The .iterrows() method makes the summary printing easy
            for _, row in op_changes.iterrows():
                print(f'  {row["old_name"]} -> {row["new_name"]}')

        print('-' * 50)
        print('Feature renaming/tagging complete. Returning modified dataframe.')
        print('-' * 50)

    elif verbose:
        print('No changes were made to the dataframe.')
        print('-' * 50)

    return df  # Return dataframe with renamed and tagged columns


def _feature_stats_core(df: pd.DataFrame, verbose: bool = True, summary_stats: bool = False) -> None:
    """
    Core function to aggregate the features by class (i.e. feature type) and print top-level missingness and
    descriptive statistics information at the feature level.
    It can build and display summary tables at the feature-class level if requested by the user.

    Args:
        df (pd.DataFrame): The input dataframe for analysis.
        verbose (bool): Whether to print more detailed information/visuals for each feature. Defaults to True.
        summary_stats (bool): Whether to print summary statistics at the feature-class level. Defaults to False.

    Returns:
        None. This is a void function which prints information to the console.
    """
    if verbose:
        print('Displaying general information for all features in dataset...')
        print('-' * 50)  # Visual separator

    # Begin with four-category feature detection process
    # Boolean features (either logical values or 0/1 integers with exactly two unique values)
    bool_cols = []
    for column in df.columns:
        # Check for explicit boolean datatypes
        if pd.api.types.is_bool_dtype(df[column]):
            bool_cols.append(column)
        # Check for 0/1 integers which function as boolean
        elif (pd.api.types.is_numeric_dtype(df[column])  # Numeric check
              and df[column].dropna().nunique() == 2  # Unique value count check
              and set(df[column].dropna().unique()).issubset({0, 1})):  # Check if unique values are a subset of {0, 1}
            bool_cols.append(column)  # If all checks are passed, this is a boolean feature

    # Datetime features
    datetime_cols = []
    for column in df.columns:
        # Check for explicit datetime datatypes
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_cols.append(column)
        # For string columns, try to parse them as datetime and assess results
        elif pd.api.types.is_object_dtype(df[column]):
            try:
                # Fetch the first non-null value to check if the parsing is possible
                first_valid = df[column].dropna().iloc[0]

                # Attempt to parse the first value as a datetime - raise ValueError if not possible
                pd.to_datetime(first_valid)

                # If we get this far, the first value is a valid datetime
                # Now we check a sample of values to confirm most are valid dates
                # I'll limit the sample to 100 values which should be enough
                sample_size = min(100, len(df[column].dropna()))

                # Take a random sample if we have values, otherwise use an empty list
                sample = df[column].dropna().sample(sample_size) if sample_size > 0 else []

                # Count how many values in the sample can be parsed as dates
                # NOTE: pd.to_datetime with errors='coerce' returns NaT for invalid dates
                valid_dates = sum(pd.to_datetime(value, errors='coerce') is not pd.NaT for value in sample)

                # Calculate the proportion of valid dates
                # I use max(1, len) to avoid dividing by zero
                valid_prop = valid_dates / max(1, len(sample))

                # If more than 80% of sampled values are valid dates, classify the feature as datetime
                # This threshold helps avoid misclassifying text fields that occasionally contain dates as datetime
                if valid_prop > 0.8:
                    datetime_cols.append(column)

            # ValueError if failure to parse first value, IndexError if the feature is empty
            except (ValueError, IndexError):
                continue

    # Categorical features (excluding those identified as datetime or boolean)
    cat_cols = [column for column in df.columns
                if (pd.api.types.is_object_dtype(df[column]) or
                    isinstance(df[column].dtype, type(pd.Categorical.dtype))) and
                column not in datetime_cols and
                column not in bool_cols]

    # Numerical features (excluding those identified as boolean)
    num_cols = [column for column in df.columns
                if pd.api.types.is_numeric_dtype(df[column]) and
                column not in bool_cols]

    # Data quality indicator arrays
    zero_var_cols = []
    near_const_cols = []
    duplicate_cols = []
    suspicious_cols = []

    # Detect zero-variance and near-constant features
    for column in df.columns:
        # Skip columns with all NaN values
        if df[column].isna().all():
            continue

        # Detect zero variance features
        if df[column].nunique() == 1:
            zero_var_cols.append(column)
            continue

        # Detect near-constant features (>95% single value)
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            value_counts = df[column].value_counts(normalize=True)
            if value_counts.iloc[0] > 0.95:
                near_const_cols.append((column, value_counts.index[0], value_counts.iloc[0] * 100))

    # Detect potentially duplicate features (using sampling for efficiency)
    # This compares the first 1000 rows of each feature for exact matches
    sample_df = df.head(1000)
    cols_to_check = df.columns

    for i, col1 in enumerate(cols_to_check):
        for col2 in cols_to_check[i + 1:]:
            # Only compare features of the same type
            if df[col1].dtype != df[col2].dtype:
                continue

            # Check if the columns are identical in the sample
            if sample_df[col1].equals(sample_df[col2]):
                duplicate_cols.append((col1, col2))

    # Display detection results if verbose mode is on
    if verbose:
        # Display feature type counts
        print(f'Feature type distribution in dataset:')
        print(f'- Boolean features: {len(bool_cols)}')
        print(f'- Datetime features: {len(datetime_cols)}')
        print(f'- Categorical features: {len(cat_cols)}')
        print(f'- Numerical features: {len(num_cols)}')
        print('-' * 50)

        # Display the names of each feature type if any exist
        if bool_cols:
            print('The boolean features are:')
            print(', '.join(bool_cols))
            print('-' * 50)
        else:
            print('No boolean features were found in the dataset.')
            print('-' * 50)

        if datetime_cols:
            print('The datetime features are:')
            print(', '.join(datetime_cols))
            print('-' * 50)
        else:
            print('No datetime features were found in the dataset.')
            print('-' * 50)

        if cat_cols:
            print('The categorical features are:')
            print(', '.join(cat_cols))
            print('-' * 50)
        else:
            print('No categorical features were found in the dataset.')
            print('-' * 50)

        if num_cols:
            print('The numerical features are:')
            print(', '.join(num_cols))
            print('-' * 50)
        else:
            print('No numerical features were found in the dataset.')
            print('-' * 50)

        # Display data quality indicators if they exist
        if zero_var_cols:
            print('ALERT: The following features have zero variance (single unique value):')
            for column in zero_var_cols:
                print(f'- {column}: {df[column].iloc[0]}')
            print('-' * 50)

        if near_const_cols:
            print('ALERT: The following features have near-constant values (>95% single value):')
            for column, value, percentage in near_const_cols:
                print(f'- {column}: value "{value}" appears in {percentage:.2f}% of non-null instances')
            print('-' * 50)

        if duplicate_cols:
            print('ALERT: The following features may be duplicates:')
            for col1, col2 in duplicate_cols:
                print(f'- {col1} and {col2}')
            print('-' * 50)

        print('Producing key values at the feature level...')

    # Enhanced helper functions for statistics
    def calculate_entropy(series):
        """Calculate Shannon entropy/information content of a series"""
        value_counts = series.value_counts(normalize=True)
        return -sum(p * np.log2(p) for p in value_counts if p > 0)

    def format_large_number(num):
        """Format large numbers with comma separators"""
        return f'{num:,}'

    def format_percentage(part, whole):
        """Calculate and format percentage"""
        if whole == 0:
            return "0.00%"
        return f'{(part / whole * 100):.2f}%'

    def show_key_vals(column: str, df: pd.DataFrame, feature_type: str):
        """This helper function calculates and prints key values and missingness info at the feature level."""
        if verbose:
            print('-' * 50)
            print(f'Key values for {feature_type} feature "{column}":')
            print('-' * 50)
        else:
            print(f'\nFeature: "{column}" - ({feature_type})')

        # Calculate missingness at feature level
        missing_cnt = df[column].isnull().sum()  # Total count
        missing_rate = (missing_cnt / len(df) * 100).round(2)  # Missingness rate
        print(f'Missing values: {format_large_number(missing_cnt)} ({missing_rate}%)')

        # Ensure the feature is not fully null before producing statistics
        if not df[column].isnull().all():
            # Boolean features statistics
            if feature_type == 'Boolean':
                value_counts = df[column].value_counts()
                true_count = int(df[column].sum())
                false_count = int(len(df[column].dropna()) - true_count)
                true_pct = (true_count / len(df[column].dropna()) * 100).round(2)

                print(f'True values: {format_large_number(true_count)} ({true_pct}%)')
                print(f'False values: {format_large_number(false_count)} ({100 - true_pct:.2f}%)')

                if verbose:
                    print('\nValue counts:')
                    print(value_counts)

            # DateTime features statistics
            elif feature_type == 'Datetime':
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    series = pd.to_datetime(df[column], errors='coerce')
                else:
                    series = df[column]

                # Get min and max dates
                min_date = series.min()
                max_date = series.max()

                # Calculate date range in days
                try:
                    range_days = (max_date - min_date).days
                    print(f'Date range: {min_date.date()} to {max_date.date()} ({range_days} days)')
                except:
                    print(f'Date range: {min_date} to {max_date}')

                if verbose:
                    # Identify common frequencies if possible
                    try:
                        # Sort data and calculate time differences
                        sorted_dates = series.dropna().sort_values()
                        time_diffs = sorted_dates.diff().dropna()

                        if len(time_diffs) > 0:
                            # Get most common time difference
                            from collections import Counter
                            counter = Counter(time_diffs.astype(str))
                            common_diff = counter.most_common(1)[0][0]

                            print(f'Most common time difference: {common_diff}')

                            # Check for regular intervals
                            if counter.most_common(1)[0][1] / len(time_diffs) > 0.5:
                                print('The data appears to have regular time intervals')
                            else:
                                print('The data has irregular time intervals')
                    except:
                        pass

                    # Calculate distribution by year, month, and day of week
                    try:
                        print('\nYear distribution:')
                        year_counts = series.dt.year.value_counts().sort_index()
                        for year, count in year_counts.items():
                            print(f'- {year}: {format_large_number(count)} ({count / len(series.dropna()) * 100:.1f}%)')

                        print('\nMonth distribution:')
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        month_counts = series.dt.month.value_counts().sort_index()
                        for month, count in month_counts.items():
                            print(
                                f'- {month_names[month - 1]}: {format_large_number(count)} ({count / len(series.dropna()) * 100:.1f}%)')

                        print('\nDay of week distribution:')
                        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                        day_counts = series.dt.dayofweek.value_counts().sort_index()
                        for day, count in day_counts.items():
                            print(
                                f'- {day_names[day]}: {format_large_number(count)} ({count / len(series.dropna()) * 100:.1f}%)')
                    except:
                        pass

            # Categorical features statistics
            elif feature_type == 'Categorical':
                value_counts = df[column].value_counts()
                unique_values = df[column].nunique()
                mode_val = df[column].mode().iloc[0] if not df[column].mode().empty else 'No mode exists'

                print(f'Unique values: {format_large_number(unique_values)}')
                print(
                    f'Mode: {mode_val} (appears {format_large_number(value_counts.iloc[0])} times, {value_counts.iloc[0] / len(df[column].dropna()) * 100:.2f}%)')

                # Add entropy calculation
                if len(df[column].dropna()) > 0:
                    entropy = calculate_entropy(df[column].dropna())
                    max_entropy = np.log2(unique_values) if unique_values > 0 else 0

                    if verbose:
                        print(f'Information entropy: {entropy:.2f} bits')
                        if max_entropy > 0:
                            print(
                                f'Normalized entropy: {entropy / max_entropy:.2f} (0=constant, 1=uniform distribution)')
                            print('\nExplanation: Entropy measures the unpredictability of the feature\'s values.')
                            print('Low entropy (near 0) means the feature is highly predictable or skewed.')
                            print('High entropy (near the maximum) means values are evenly distributed.')

                if verbose:
                    print('\nValue counts (top 10):')
                    print(value_counts.head(10))

                    if len(value_counts) > 10:
                        print(f'...and {len(value_counts) - 10} more values')

                    # Top frequency ratios
                    if len(value_counts) >= 2:
                        print('\nTop to second frequency ratio: {:.2f}'.format(
                            value_counts.iloc[0] / value_counts.iloc[1]))

                    # Show distribution pattern
                    if len(value_counts) > 5:
                        print('\nDistribution pattern:')
                        total = len(df[column].dropna())
                        cumulative = 0
                        for i in range(min(5, len(value_counts))):
                            v = value_counts.iloc[i]
                            p = v / total * 100
                            cumulative += p
                            print(f'- Top {i + 1} values cover: {cumulative:.1f}% of data')

            # Numerical features statistics
            elif feature_type == 'Numerical':
                stats = df[column].describe()

                # Basic statistics
                print(f'Mean: {stats["mean"]:.4f}')
                print(f'Min: {stats["min"]:.4f}')
                print(f'Max: {stats["max"]:.4f}')

                # Enhanced statistics when verbose is True
                if verbose:
                    print(f'Median: {stats["50%"]:.4f}')
                    print(f'Std Dev: {stats["std"]:.4f}')

                    # Add quartile information
                    print(f'25th percentile: {stats["25%"]:.4f}')
                    print(f'75th percentile: {stats["75%"]:.4f}')
                    print(f'IQR: {(stats["75%"] - stats["25%"]):.4f}')

                    # Add skewness
                    skew = df[column].skew()
                    print(f'Skewness: {skew:.4f}')
                    if abs(skew) < 0.5:
                        print('  (approximately symmetric distribution)')
                    elif abs(skew) < 1:
                        print('  (moderately skewed distribution)')
                    else:
                        print('  (highly skewed distribution)')

                    # Add kurtosis
                    kurt = df[column].kurtosis()
                    print(f'Kurtosis: {kurt:.4f}')
                    if kurt < -0.5:
                        print('  (platykurtic - flatter than normal distribution)')
                    elif kurt > 0.5:
                        print('  (leptokurtic - more peaked than normal distribution)')
                    else:
                        print('  (mesokurtic - similar to normal distribution)')

                    # Coefficient of variation
                    if stats["mean"] != 0:
                        cv = stats["std"] / stats["mean"]
                        print(f'Coefficient of variation: {cv:.4f}')

                    # Add brief explanations
                    print('\nExplanation of statistics:')
                    print('- Skewness: Measures asymmetry of the distribution')
                    print('- Kurtosis: Measures "tailedness" of the distribution')
                    print('- IQR: Interquartile range (middle 50% of data)')
                    if stats["mean"] != 0:
                        print('- Coefficient of variation: Relative standard deviation')

                        # Display feature-level statistics

    if bool_cols:
        if verbose:
            print('\nKEY VALUES FOR BOOLEAN FEATURES:')
        for column in bool_cols:
            show_key_vals(column, df, 'Boolean')

    if datetime_cols:
        if verbose:
            print('\nKEY VALUES FOR DATETIME FEATURES:')
        for column in datetime_cols:
            show_key_vals(column, df, 'Datetime')

    if cat_cols:
        if verbose:
            print('\nKEY VALUES FOR CATEGORICAL FEATURES:')
        for column in cat_cols:
            show_key_vals(column, df, 'Categorical')

    if num_cols:
        if verbose:
            print('\nKEY VALUES FOR NUMERICAL FEATURES:')
        for column in num_cols:
            show_key_vals(column, df, 'Numerical')

    # Display feature-class level summary statistics if requested
    if summary_stats:
        if bool_cols:
            if verbose:
                print('\nBOOLEAN FEATURES SUMMARY STATISTICS:')
                print('-' * 50)

            bool_summary = pd.DataFrame({
                'True_count': [df[column].sum() for column in bool_cols],
                'True_rate': [(df[column].sum() / len(df[column].dropna()) * 100).round(2)
                              for column in bool_cols],
                'Missing_count': [df[column].isnull().sum() for column in bool_cols],
                'Missing_rate': [(df[column].isnull().sum() / len(df) * 100).round(2)
                                 for column in bool_cols]
            }, index=bool_cols)
            print('Boolean Features Summary Table:')
            print(str(bool_summary))

        if datetime_cols:
            if verbose:
                print('\nDATETIME FEATURES SUMMARY STATISTICS:')
                print('-' * 50)

            # Convert any string datetime columns temporarily for statistics
            datetime_data = {}
            for column in datetime_cols:
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    series = pd.to_datetime(df[column], errors='coerce')
                else:
                    series = df[column]

                min_date = series.min()
                max_date = series.max()
                try:
                    range_days = (max_date - min_date).days
                except:
                    range_days = np.nan

                datetime_data[column] = {
                    'min_date': min_date,
                    'max_date': max_date,
                    'range_days': range_days,
                    'missing_count': df[column].isnull().sum(),
                    'missing_rate': (df[column].isnull().sum() / len(df) * 100).round(2)
                }

            # Create summary dataframe
            datetime_summary = pd.DataFrame({
                'Min_date': [datetime_data[col]['min_date'] for col in datetime_cols],
                'Max_date': [datetime_data[col]['max_date'] for col in datetime_cols],
                'Range_days': [datetime_data[col]['range_days'] for col in datetime_cols],
                'Missing_count': [datetime_data[col]['missing_count'] for col in datetime_cols],
                'Missing_rate': [datetime_data[col]['missing_rate'] for col in datetime_cols]
            }, index=datetime_cols)

            print('Datetime Features Summary Table:')
            print(str(datetime_summary))

        if cat_cols:
            if verbose:
                print('\nCATEGORICAL FEATURES SUMMARY STATISTICS:')
                print('-' * 50)

            cat_summary = pd.DataFrame({
                'Unique_values': [df[column].nunique() for column in cat_cols],
                'Mode_frequency': [df[column].value_counts().iloc[0] if len(df[column].value_counts()) > 0 else 0
                                   for column in cat_cols],
                'Mode_percentage': [(df[column].value_counts().iloc[0] / len(df[column].dropna()) * 100).round(2)
                                    if len(df[column].value_counts()) > 0 else 0
                                    for column in cat_cols],
                'Entropy': [calculate_entropy(df[column].dropna()) if len(df[column].dropna()) > 0 else 0
                            for column in cat_cols],
                'Missing_count': [df[column].isnull().sum() for column in cat_cols],
                'Missing_rate': [(df[column].isnull().sum() / len(df) * 100).round(2)
                                 for column in cat_cols]
            }, index=cat_cols)
            print('Categorical Features Summary Table:')
            print(str(cat_summary))

        if num_cols:
            if verbose:
                print('\nNUMERICAL FEATURES SUMMARY STATISTICS:')
                print('-' * 50)

            # Enhanced numerical summary with additional statistics
            num_summary = df[num_cols].describe()

            # Add skewness and kurtosis
            num_summary.loc['skew'] = df[num_cols].skew()
            num_summary.loc['kurt'] = df[num_cols].kurtosis()

            # Add coefficient of variation where mean is not zero
            cv_row = []
            for col in num_cols:
                mean_val = df[col].mean()
                std_val = df[col].std()
                cv_row.append(std_val / mean_val if mean_val != 0 else np.nan)

            num_summary.loc['cv'] = cv_row

            # Add missing value count and percentage
            num_summary.loc['missing'] = df[num_cols].isnull().sum()
            num_summary.loc['missing %'] = (df[num_cols].isnull().sum() / len(df) * 100).round(2)

            print('Numerical Features Summary Table:')
            print(str(num_summary))

    if verbose:
        print('-' * 50)
        print('Feature statistics analysis complete.')
        print('-' * 50)


def _impute_core(df: pd.DataFrame, verbose: bool = True, skip_warnings: bool = False) -> pd.DataFrame:
    """
    Core function to perform simple imputation for missing values at the feature level.
    Supports mean, median, mode, constant value, random sampling, and time-series imputation methods
    based on feature type and data characteristics.

    Args:
        df (pd.DataFrame): Input DataFrame containing features to impute.
        verbose (bool): Whether to display detailed guidance and explanations. Defaults to True.
        skip_warnings (bool): Whether to skip missingness threshold warnings. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with imputed values where specified by user.

    Raises:
        ValueError: If an invalid imputation method is selected
    """
    def plot_dist_comps(original: pd.Series, imputed: pd.Series, feature_name: str) -> None:
        """
        This helper function creates and displays a side-by-side comparison of pre- and post- imputation distributions.

        Args:
            original: Original series with missing values
            imputed: Series after imputation
            feature_name: Name of the feature being visualized
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

            # Pre-imputation distribution
            sns.histplot(data=original.dropna(), ax=ax1)
            ax1.set_title(f'Pre-imputation Distribution of "{feature_name}"')
            ax1.set_xlabel(feature_name)

            # Post-imputation distribution
            sns.histplot(data=imputed, ax=ax2)
            ax2.set_title(f'Post-imputation Distribution of "{feature_name}"')
            ax2.set_xlabel(feature_name)

            plt.tight_layout()
            plt.show()
            plt.close()

        except Exception as exc:
            print(f'Could not create distribution visualizations: {str(exc)}')
            if plt.get_fignums():
                plt.close('all')

    def low_variance_check(series: pd.Series, threshold: float = 0.01) -> bool:
        """
        This helper function checks if a series has a variance below the specified threshold.

        Args:
            series: Numerical series to check
            threshold: Variance threshold, default 0.01

        Returns:
            bool: True if variance is below threshold
        """
        return series.var() < threshold

    def outlier_cnt_check(series: pd.Series, threshold: float = 0.1) -> bool:
        """
        This helper function checks if a series has more outliers than the specified threshold.

        Args:
            series: Numerical series to check
            threshold: Maximum allowable proportion of outliers, default 0.1

        Returns:
            bool: True if proportion of outliers exceeds threshold
        """
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = series < (q1 - 1.5 * iqr)
        upper_bound = series > (q3 + 1.5 * iqr)
        outlier_mask = pd.Series(lower_bound | upper_bound, index=series.index)
        return (outlier_mask.sum() / len(series)) > threshold

    def high_corr_check(df: pd.DataFrame, columns: list[str], threshold: float = 0.9) -> list[tuple]:
        """
        This helper function identifies pairs of highly correlated numerical features.

        Args:
            df: DataFrame containing the features
            columns: List of numerical column names to check
            threshold: Correlation threshold, default 0.9

        Returns:
            list[tuple]: List of tuples containing (col1, col2, correlation)
        """
        if len(columns) < 2:
            return []

        corr_matrix = df[columns].corr()
        high_corr_pairs = []

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((
                        columns[i],
                        columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        return high_corr_pairs

    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning imputation process.')
        print('-' * 50)  # Visual separator

    # Check if there are no missing values - if no missing values exist, skip imputation
    if not df.isnull().any().any():
        print('WARNING: No missing values are present in dataset. Skipping imputation. Dataframe was not modified.')
        return df

    # Identify initial feature types using list comprehensions
    cat_cols = [column for column in df.columns if pd.api.types.is_object_dtype(df[column])
                or isinstance(df[column].dtype, type(pd.Categorical.dtype))]

    num_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

    # Check for presence of datetime columns and index
    datetime_cols = []

    for column in df.columns:
        # Check if column is already datetime type
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            datetime_cols.append(column)
        # For string columns, try to parse as datetime
        elif pd.api.types.is_object_dtype(df[column]):
            try:
                # Try to parse first non-null value
                first_valid = df[column].dropna().iloc[0]
                pd.to_datetime(first_valid)
                datetime_cols.append(column)
            except (ValueError, IndexError):
                continue

    is_datetime_index = pd.api.types.is_datetime64_any_dtype(df.index)
    has_datetime = bool(datetime_cols or is_datetime_index)

    # Convert detected datetime string columns to datetime type
    if datetime_cols:
        for col in datetime_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except ValueError:
                    datetime_cols.remove(col)  # Remove if conversion fails

    # Top-level feature classification information if Verbose is True
    if verbose:
        print('Initial feature type identification:')
        print(f'Categorical features: {", ".join(cat_cols) if cat_cols else "None"}')
        print(f'Numerical features: {", ".join(num_cols) if num_cols else "None"}')
        if has_datetime:
            print('ALERT: Datetime elements detected in dataset.')
        print('-' * 50)

    # Check for numeric features that might actually be categorical in function/intent
    true_nums = num_cols.copy()  # Start with all numeric columns
    for column in num_cols:
        unique_vals = sorted(df[column].dropna().unique())

        # We suspect that any all-integer column with five or fewer unique values is actually categorical
        if len(unique_vals) <= 5 and all(float(x).is_integer() for x in unique_vals if pd.notna(x)):
            if verbose:
                print(f'Feature "{column}" has only {len(unique_vals)} unique integer values: '
                      f'{[int(val) for val in unique_vals]}')
                print('ALERT: This could be a categorical feature encoded as numbers, e.g. a 1/0 representation of '
                      'Yes/No values.')

            # Ask the user to assess and reply
            user_cat = input(f'\nShould "{column}" actually be treated as categorical? (Y/N): ')

            # If user agrees, recast the feature to string-type and append to list of categorical features
            if user_cat.lower() == 'y':
                df[column] = df[column].apply(lambda value: str(value) if pd.notna(value) else value)
                cat_cols.append(column)
                true_nums.remove(column)  # Remove from numeric if identified as categorical
                if verbose:
                    print(f'Converted numerical feature "{column}" to categorical type.')
                    print('-' * 50)

    # Check input dataframe's numerical features for potential data quality problems
    qual_probs = {}
    for column in true_nums:
        problems = []

        if low_variance_check(df[column]):
            problems.append('Near-zero variance')

        if outlier_cnt_check(df[column]):
            problems.append('High outlier count')

        if abs(df[column].skew()) > 2:
            problems.append('High skewness')

        if problems:
            qual_probs[column] = problems

    high_corr_pairs = high_corr_check(df, true_nums)

    # Print data quality warnings if warnings are not skipped
    if not skip_warnings and (qual_probs or high_corr_pairs):
        print('-' * 50)
        print('WARNING: POTENTIAL DATA QUALITY ISSUES DETECTED!')
        print('-' * 50)

        if qual_probs:
            print('Feature-specific issues:')
            for col, issues in qual_probs.items():
                print(f'- {col}: {", ".join(issues)}')

        if high_corr_pairs:
            print('\nHighly correlated feature pairs:')
            for col1, col2, corr in high_corr_pairs:
                print(f'- {col1} & {col2}: correlation = {corr:.2f}')

        if verbose:
            print('\nNOTE: These issues may affect imputation quality:')
            print('- Near-zero variance: Feature may not be usefully informative.')
            print('- High outlier count: Mean-substitution imputation may be inappropriate.')
            print('- High correlation: Features may contain redundant information.')
            print('- High skewness: Transforming data may be necessary prior to imputation.')
        print('-' * 50)

    # Final feature classification info if Verbose is True
    if verbose:
        print('Final feature type classification:')
        print(f'Categorical features: {", ".join(cat_cols) if cat_cols else "None"}')
        print(f'Numerical features: {", ".join(true_nums) if true_nums else "None"}')
        print('-' * 50)

    # For each feature, calculate missingness statistics
    if verbose:
        print('Count and rate of missingness for each feature:')

    # Calculate missingness values
    missingness_vals = {}
    for column in df.columns:
        missing_cnt = df[column].isnull().sum()
        missing_rate = (missing_cnt / len(df) * 100).round(2)
        missingness_vals[column] = {'count': missing_cnt, 'rate': missing_rate}

    if verbose:
        for column, vals in missingness_vals.items():
            print(f'Feature {column} has {vals["count"]} missing value(s). ({vals["rate"]}% missing)')

    # Check if this is time series data if datetime elements exist
    is_timeseries = False  # Assume the data are not timeseries as the naive case
    if has_datetime:
        if verbose:
            print('\nALERT: Datetime elements detected in the dataset.')
            print('Time series data may benefit from forward/backward fill imputation.')

        ts_response = input('Are these time series data? (Y/N): ').lower()
        is_timeseries = ts_response == 'y'

        if verbose and is_timeseries:
            print('\nTime series imputation methods available in TADPREP:')
            print('- Forward fill: Carries the last valid observation forward')
            print('- Backward fill: Carries the next valid observation backward')
            print('-' * 50)

    if not skip_warnings and verbose:
        print('\nWARNING: Imputing missing values for features with a missing rate over 10% is not recommended '
              'due to potential bias introduction.')

    # Build list of good candidates for imputation based on missingness and data quality
    if not skip_warnings:
        imp_candidates = [
            key for key, value in missingness_vals.items()
            # Check for missingness problems
            if 0 < value['rate'] <= 10]

    else:
        imp_candidates = [key for key, value in missingness_vals.items() if 0 < value['rate']]

    # We only walk through the imputation missingness-rate guidance if warnings aren't skipped
    if not skip_warnings:
        if imp_candidates and verbose:
            print('Based on missingness rates and data quality assessments, the following features are good '
                  'candidates for imputation:')

            for key in imp_candidates:
                print(f'- {key}: {missingness_vals[key]["rate"]}% missing')

        elif verbose:
            print('No features fall within the recommended criteria for imputation.')
            print('WARNING: Statistical best practices indicate you should not perform imputation.')

    # Store imputation records for summary
    imp_records = []

    # Ask if user wants to override the recommendation
    while True:
        try:
            user_override = input('\nDo you wish to:\n'
                                  '1. Impute only for recommended features (<= 10% missing)\n'
                                  '2. Override the warning and consider all features with missing values\n'
                                  '3. Skip imputation\n'
                                  'Enter choice: ').strip()

            # Catch bad input
            if user_override not in ['1', '2', '3']:
                raise ValueError('Please enter 1, 2, or 3.')

            if user_override == '3':
                print('Skipping imputation. No changes made to dataset.')
                return df

            # Build list of features to be imputed
            imp_features = (imp_candidates if user_override == '1'
                            else [key for key, value in missingness_vals.items() if value['count'] > 0])

            # If user wants to follow missingness guidelines and no good candidates exist, skip imputation
            if not imp_features:
                print('No features available for imputation given user input. Skipping imputation.')
                return df

            # Exit the loop once the imp_features list is built
            break

        # Catch all other input errors
        except ValueError as exc:
            print(f'Invalid input: {exc}')
            continue  # Restart loop

    # Offer methodology explanations if desired by user if Verbose is True
    if verbose:
        print('\nWARNING: TADPREP supports the following imputation methods:')
        print('- Mean/Median/Mode imputation')
        print('- Constant value imputation')
        print('- Random sampling from non-null within-feature values')

        if is_timeseries:
            print('- Forward/backward fill (for time series data)')

        print('\nFor more sophisticated methods (e.g. imputation-by-modeling), skip this step and write '
              'your own imputation code.')

        user_impute_refresh = input('Do you want to see a brief refresher on these imputation methods? (Y/N): ')
        if user_impute_refresh.lower() == 'y':
            print('\nImputation Methods Overview:')
            print('Statistical Methods:')
            print('- Mean: Best for normally-distributed data. Theoretically practical. Sensitive to outliers.')
            print('- Median: Better for skewed numerical data. Robust to outliers.')
            print('- Mode: Useful for categorical and fully-discrete numerical data.')
            print('\nOther Methods:')
            print('- Constant: Replaces missing values with a specified value. Good when default values exist.')
            print('- Random Sampling: Maintains feature distribution by sampling from non-null values.')

            if is_timeseries:
                print('\nTime Series Imputation Methods:')
                print('- Forward Fill: Carries last valid value forward. Good for continuing trends.')
                print('- Backward Fill: Carries next valid value backward. Good for establishing history.')

    # Begin imputation at feature level
    for feature in imp_features:
        print(f'\nProcessing feature "{feature}"...')
        print('-' * 50)
        if verbose:
            print(f'- Datatype: {df[feature].dtype}')
            print(f'- Missingness rate: {missingness_vals[feature]["rate"]}%')

            # Show pre-imputation distribution if numerical
            if feature in true_nums:
                print('\nPre-imputation distribution:')
                print(df[feature].describe().round(2))

                # Offer visualization
                if input('\nWould you like to view a plot of the current feature distribution? (Y/N): ').lower() == 'y':
                    try:
                        plt.figure(figsize=(12, 8))
                        sns.histplot(data=df, x=feature)
                        plt.title(f'Distribution of {feature} (Pre-imputation)')
                        plt.show()
                        plt.close()

                    except Exception as exc:
                        print(f'Could not create visualization: {exc}')
                        if plt.get_fignums():
                            plt.close('all')

        # Build list of available/valid imputation methods based on feature characteristics
        val_methods = ['Skip imputation for this feature']

        if feature in true_nums:
            val_methods = ['Mean', 'Median', 'Mode', 'Constant', 'Random Sample'] + val_methods

        else:
            val_methods = ['Mode', 'Constant', 'Random Sample'] + val_methods

        if is_timeseries:
            val_methods = ['Forward Fill', 'Backward Fill'] + val_methods

        # Prompt user to select an imputation method
        while True:
            method_items = [f'{idx}. {method}' for idx, method in enumerate(val_methods, 1)]
            method_prompt = f'\nChoose imputation method:\n{"\n".join(method_items)}\nEnter choice: '
            user_imp_choice = input(method_prompt)

            try:
                method_idx = int(user_imp_choice) - 1
                if 0 <= method_idx < len(val_methods):
                    imp_method = val_methods[method_idx]
                    break
                else:
                    print('Invalid input. Enter a valid number.')
            except ValueError:
                print('Invalid input. Enter a valid number.')

        # Exit current feature if user chooses to skip
        if imp_method == 'Skip imputation for this feature':
            if verbose:
                print(f'Skipping imputation for feature: "{feature}"')
            continue

        # Begin actual imputation process
        try:
            # Store original values for distribution comparison
            original_values = df[feature].copy()
            feature_missing_cnt = missingness_vals[feature]['count']

            # Calculate imputation values based on method selection
            if imp_method == 'Mean':
                imp_val = df[feature].mean()
                df[feature] = df[feature].fillna(imp_val)
                method_desc = f'Mean value ({imp_val:.4f})'

            elif imp_method == 'Median':
                imp_val = df[feature].median()
                df[feature] = df[feature].fillna(imp_val)
                method_desc = f'Median value ({imp_val:.4f})'

            elif imp_method == 'Mode':
                mode_vals = df[feature].mode()
                if len(mode_vals) == 0:
                    print(f'No mode value exists for feature {feature}. Skipping imputation.')
                    continue

                imp_val = mode_vals[0]  # Select first mode

                df[feature] = df[feature].fillna(imp_val)
                method_desc = f'Mode value ({imp_val})'

            elif imp_method == 'Constant':
                while True:
                    try:
                        if feature in true_nums:
                            user_input = input('Enter constant numerical value for imputation: ')
                            imp_val = float(user_input)

                        else:
                            # For categorical features, show existing categories
                            existing_cats = df[feature].dropna().unique()
                            print('\nExisting categories in feature:')
                            for idx, cat in enumerate(existing_cats, 1):
                                print(f'{idx}. {cat}')

                            user_input = input('\nEnter the number of the category to use for imputation: ').strip()

                            # Validate category selection
                            try:
                                cat_idx = int(user_input) - 1
                                if 0 <= cat_idx < len(existing_cats):
                                    imp_val = existing_cats[cat_idx]

                                else:
                                    raise ValueError('Selected category number is out of range')

                            except ValueError:
                                raise ValueError('Please enter a valid category number')

                        break  # Exit loop if input is valid

                    except ValueError as exc:
                        print(f'Invalid input: {str(exc)}. Please try again.')

                df[feature] = df[feature].fillna(imp_val)
                method_desc = f'Constant value ({imp_val})'

            elif imp_method == 'Random Sample':
                # Get non-null values to sample from
                valid_values = df[feature].dropna()

                if len(valid_values) == 0:
                    print('No valid values to sample from. Skipping imputation.')
                    continue

                imp_vals = valid_values.sample(n=feature_missing_cnt, replace=True)  # Sample with replacement

                df.loc[df[feature].isna(), feature] = imp_vals.values  # Fill NaN values with sampled values

                method_desc = 'Random sampling from non-null values'

            elif imp_method == 'Forward Fill':
                df[feature] = df[feature].ffill()
                method_desc = 'Forward fill'

            elif imp_method == 'Backward Fill':
                df[feature] = df[feature].bfill()
                method_desc = 'Backward fill'

            # Add record for summary table
            imp_records.append({
                'feature': feature,
                'count': feature_missing_cnt,
                'method': imp_method,
                'description': method_desc
            })

            if verbose:
                print(f'\nSuccessfully imputed {feature_missing_cnt} missing values using {method_desc}.')

                # Show post-imputation distribution for numerical features
                if feature in true_nums:
                    print('\nPost-imputation distribution:')
                    print(df[feature].describe().round(2))

                    if input('\nWould you like to see comparison plots of the pre- and post-imputation feature '
                             'distribution? (Y/N): ').lower() == 'y':
                        plot_dist_comps(original_values, df[feature], feature)

        except Exception as exc:
            print(f'Error during imputation for feature {feature}: {exc}')
            print('Skipping imputation for this feature.')
            continue

    # Print imputation summary if any features were imputed
    if verbose:
        if imp_records:
            print('\nIMPUTATION SUMMARY:')
            print('-' * 50)

            for record in imp_records:
                print(f'Feature: {record["feature"]}')
                print(f'- {record["count"]} values imputed')
                print(f'- Method: {record["method"]}')
                print(f'- Description: {record["description"]}')
                print('-' * 50)

        print('Imputation complete. Returning modified dataframe.')
        print('-' * 50)

    return df  # Return the modified dataframe with imputed values


def _encode_core(
        df: pd.DataFrame,
        features_to_encode: list[str] | None = None,
        verbose: bool = True,
        skip_warnings: bool = False
) -> pd.DataFrame:
    """
    Core function to encode categorical features using one-hot or dummy encoding, as specified by user.
    The function also looks for false-numeric features (e.g. 1/0 representations of 'Yes'/'No') and asks if they
    should be treated as categorical and therefore be candidates for encoding.

    Args:
        df : pd.DataFrame
            Input DataFrame containing features to encode.
        features_to_encode : list[str] | None, default=None
            Optional list of features to encode. If None, function will help the user identify categorical features.
        verbose : bool, default=True
            Whether to display detailed guidance and explanations.
        skip_warnings : bool, default=False
            Whether to skip all best-practice-related warnings about null values and cardinality issues.

    Returns:
        pd.DataFrame: DataFrame with encoded values as specified by user. Original features are dropped after encoding.

    Raises:
        ValueError: If all selected features have already been encoded
        ValueError: If encoding fails due to data structure issues
    """
    if verbose:
        print('-' * 50)  # Visual separator
        print('Beginning encoding process.')
        print('-' * 50)  # Visual separator

    if verbose and features_to_encode:
        print('Encoding only features in user-provided list:')
        print(features_to_encode)
        print('-' * 50)  # Visual separator

    # If no features are specified in the to_encode list, identify potential categorical features
    if features_to_encode is None:
        # Identify obvious categorical features (i.e. those which are object or categorical in data-type)
        cat_cols = [column for column in df.columns
                    if pd.api.types.is_object_dtype(df[column]) or
                    isinstance(df[column].dtype, type(pd.Categorical.dtype))]

        # Check for numeric features which might actually be categorical in function/role
        numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

        for column in numeric_cols:
            unique_vals = sorted(df[column].dropna().unique())  # Get sorted unique values excluding nulls

            # We suspect that any all-integer column with five or fewer unique values is actually categorical
            if len(unique_vals) <= 5 and all(float(x).is_integer() for x in unique_vals if pd.notna(x)):

                # Ask user to assess and reply
                if verbose:
                    print(f'Feature "{column}" has only {len(unique_vals)} unique integer values: '
                          f'{[int(val) for val in unique_vals]}')
                    print('ALERT: This could be a categorical feature encoded as numbers, e.g. a 1/0 representation of '
                          'Yes/No values.')

                user_cat = input(f'Should "{column}" actually be treated as categorical? (Y/N): ')

                if user_cat.lower() == 'y':  # If the user says yes
                    df[column] = df[column].astype(str)  # Convert to string type for encoding
                    cat_cols.append(column)  # Add that feature to list of categorical features
                    if verbose:
                        print('-' * 50)
                        print(f'Converted numerical feature "{column}" to categorical type.')
                        print(f'"{column}" is now a candidate for encoding.')
                        print('-' * 50)

        final_cat_cols = cat_cols

    else:
        final_cat_cols = features_to_encode

    # Validate that all specified features exist in the dataframe
    missing_cols = [column for column in final_cat_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f'Features not found in dataframe: {missing_cols}')

    if not final_cat_cols:
        if verbose:
            print('No features were identified as candidates for encoding.')
            print('-' * 50)  # Visual separator
        print('Skipping encoding. Dataset was not modified.')
        return df

    # Instantiate empty lists for encoded data and tracking
    encoded_dfs = []  # Will hold encoded DataFrames for each feature
    columns_to_drop = []  # Will track original features to be dropped after encoding
    encoded_features = []  # Will track features and their encoding methods for reporting

    # Offer explanation of encoding methods if in verbose mode
    if verbose:
        user_encode_refresh = input('Would you like to see an explanation of encoding methods? (Y/N): ')
        if user_encode_refresh.lower() == 'y':
            print('\nOverview of One-Hot vs. Dummy Encoding:'
                  '\nOne-Hot Encoding: '
                  '\n- Creates a new binary column for every unique category.'
                  '\n- No information is lost, which preserves interpretability, but more features are created.'
                  '\n- This method is preferred for non-linear models like decision trees.'
                  '\n'
                  '\nDummy Encoding:'
                  '\n- Creates n-1 binary columns given n categories in the feature.'
                  '\n- One category becomes the reference class, and is represented by all zeros.'
                  '\n- Dummy encoding is preferred for linear models, as it avoids perfect multi-collinearity.'
                  '\n- This method is more computationally- and space-efficient, but is less interpretable.')

    if verbose:
        print('\nFinal candidate features for encoding are:')
        print(final_cat_cols)
        print()

    # Print verbose reminder about not encoding target features
    if features_to_encode is None and verbose:
        print('REMINDER: Target features (prediction targets) should not be encoded.')
        print('If any of the identified encodable features are targets, do not encode them.')

    # Process each feature in our list
    for column in final_cat_cols:
        if verbose:
            print('-' * 50)  # Visual separator
        print(f'Processing feature: "{column}"')
        if verbose:
            print('-' * 50)  # Visual separator

        # Check for nulls if warnings aren't suppressed
        null_count = df[column].isnull().sum()
        if null_count > 0 and not skip_warnings:
            # Check to see if user wants to proceed
            print(f'Warning: "{column}" contains {null_count} null values.')
            if input('Continue encoding this feature? (Y/N): ').lower() != 'y':
                continue

        # Perform cardinality checks if warnings aren't suppressed
        unique_count = df[column].nunique()
        if not skip_warnings:
            # Check for high cardinality
            if unique_count > 20:
                print(f'Warning: "{column}" has high cardinality ({unique_count} unique values)')
                if verbose:
                    print('Consider using dimensionality reduction techniques instead of encoding this feature.')
                    print('Encoding high-cardinality features can lead to issues with the curse of dimensionality.')
                # Check to see if user wants to proceed
                if input('Continue encoding this feature? (Y/N): ').lower() != 'y':
                    continue

            # Skip constant features (those with only one unique value)
            elif unique_count <= 1:
                print(f'Skipping encoding for "{column}".')
                if verbose:
                    print(f'"{column}" has only one unique value and thus provides no meaningful information.')
                continue

            # Check for low-frequency categories
            value_counts = df[column].value_counts()
            low_freq_cats = value_counts[value_counts < 10]  # Categories with fewer than 10 instances
            if not low_freq_cats.empty:
                if verbose:
                    print(f'\nWarning: Found {len(low_freq_cats)} categories with fewer than 10 instances:')
                    print(low_freq_cats)
                print('Consider grouping rare categories before encoding.')
                if input('Continue encoding this feature? (Y/N): ').lower() != 'y':
                    continue

        if verbose:
            # Show current value distribution
            print(f'\nCurrent values in "{column}":')
            print(df[column].value_counts())

            # Offer distribution plot
            if input('\nWould you like to see a plot of the value distribution? (Y/N): ').lower() == 'y':
                try:
                    plt.figure(figsize=(12, 10))
                    value_counts = df[column].value_counts()
                    plt.bar(range(len(value_counts)), value_counts.values)
                    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
                    plt.title(f'Distribution of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Count')
                    plt.tight_layout()
                    plt.show()
                    plt.close()  # Explicitly close the figure

                # Catch plotting errors
                except Exception as exc:
                    print(f'Error creating plot: {str(exc)}')
                    if plt.get_fignums():  # If any figures are open
                        plt.close('all')  # Close all figures

        if verbose:
            # Fetch encoding method preference from user
            print('\nEncoding methods:')

        # Show encoding options (needed regardless of verbosity)
        print('1. One-Hot Encoding (new column for each category)')
        print('2. Dummy Encoding (n-1 columns, drops first category)')
        print('3. Skip encoding for this feature')

        while True:
            method = input('Select encoding method or skip encoding (Enter 1, 2 or 3): ')
            if method in ['1', '2', '3']:
                break
            print('Invalid choice. Please enter 1, 2, or 3.')

        try:

            # Skip encoding if user enters the skip option
            if method == '3':
                if verbose:
                    print(f'\nSkipping encoding for feature "{column}".')
                continue

            # Apply selected encoding method
            if method == '1':
                # One-hot encoding creates a column for every category
                encoded = pd.get_dummies(df[column], prefix=column, prefix_sep='_')
                encoded_features.append(f'{column} (One-Hot)')

            else:
                # Dummy encoding creates n-1 columns
                encoded = pd.get_dummies(df[column], prefix=column, prefix_sep='_', drop_first=True)
                encoded_features.append(f'{column} (Dummy)')

            # Append encoded df and add feature to list of features to drop from the df
            encoded_dfs.append(encoded)
            columns_to_drop.append(column)

            # Note successful encoding action
            if verbose:
                print(f'\nSuccessfully encoded "{column}".')

        # Catch all other errors
        except Exception as exc:
            print(f'Error encoding feature "{column}": {str(exc)}')
            continue

    # Apply all encodings at once if any were successful
    if encoded_dfs:
        df = df.drop(columns=columns_to_drop)  # Remove original columns
        df = pd.concat([df] + encoded_dfs, axis=1)  # Add encoded columns

        # Print summary of encoding if in verbose mode
        if verbose:
            print('\nENCODING SUMMARY:')
            for feature in encoded_features:
                print(f'- {feature}')

    if verbose:
        print('-' * 50)  # Visual separator
        print('Encoding complete. Returning modified dataframe.')
        print('-' * 50)  # Visual separator

    # Return the modified dataframe with encoded values
    return df


def _scale_core(
        df: pd.DataFrame,
        features_to_scale: list[str] | None = None,
        verbose: bool = True,
        skip_warnings: bool = False
) -> pd.DataFrame:
    """
    Core function to scale numerical features using standard, robust, or minmax scaling methods.

    Args:
        df : pd.DataFrame
            Input DataFrame containing features to scale.
        features_to_scale : list[str] | None, default=None
            Optional list of features to scale. If None, function will help identify numerical features.
        verbose : bool, default=True
            Whether to display detailed guidance and explanations.
        skip_warnings : bool, default=False
            Whether to skip all best-practice-related warnings about nulls, outliers, etc.

    Returns:
        pd.DataFrame: DataFrame with scaled values as specified by user.

    Raises:
        ValueError: If scaling fails due to data structure issues
    """
    if verbose:
        print('-' * 50)
        print('Beginning scaling process.')
        print('-' * 50)

    if verbose and features_to_scale:
        print('Scaling only features in user-provided list:')
        print(features_to_scale)
        print('-' * 50)

    # If no features specified, identify potential numerical features
    if features_to_scale is None:
        # Identify all numeric features
        numeric_cols = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]

        # Check for numeric features which might actually be categorical
        for column in numeric_cols:
            unique_vals = sorted(df[column].dropna().unique())

            # We suspect any all-integer column with five or fewer unique values is categorical
            if len(unique_vals) <= 5 and all(float(x).is_integer() for x in unique_vals if pd.notna(x)):
                if verbose:
                    print(f'\nFeature "{column}" has only {len(unique_vals)} unique integer values: '
                          f'{[int(val) for val in unique_vals]}')
                    print('ALERT: This could be a categorical feature encoded as numbers, '
                          'e.g. a 1/0 representation of Yes/No values.')

                user_cat = input(f'Should "{column}" be treated as categorical and excluded from scaling? (Y/N): ')
                if user_cat.lower() == 'y':
                    numeric_cols.remove(column)
                    if verbose:
                        print(f'Excluding "{column}" from scaling consideration.')
                        print('-' * 50)

        final_numeric_cols = numeric_cols

    else:
        final_numeric_cols = features_to_scale

    # Validate that all specified features exist in the dataframe
    missing_cols = [column for column in final_numeric_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f'Features not found in DataFrame: {missing_cols}')

    if not final_numeric_cols:
        if verbose:
            print('No features were identified as candidates for scaling.')
            print('-' * 50)
        print('Skipping scaling. Dataset was not modified.')
        return df

    # Print reminder about not scaling target features
    if features_to_scale is None and verbose:
        print('REMINDER: Target features (prediction targets) should not be scaled.')
        print('If any of the identified features are targets, do not scale them.')
        print()

    # Track scaled features for reporting
    scaled_features = []

    # Offer explanation of scaling methods if verbose
    if verbose:
        user_scale_refresh = input('Would you like to see an explanation of scaling methods? (Y/N): ')
        if user_scale_refresh.lower() == 'y':
            print('\nOverview of the Standard, Robust, and MinMax Scalers:'
                  '\nStandard Scaler (Z-score normalization):'
                  '\n- Transforms features to have zero mean and unit variance.'
                  '\n- Best choice for comparing measurements in different units.'
                  '\n- Good for methods that assume normally distributed data.'
                  '\n- Not ideal when data has many outliers.'
                  '\n'
                  '\nRobust Scaler (Uses median and IQR):'
                  '\n- Scales using statistics that are resistant to outliers.'
                  '\n- Great for data where outliers are meaningful.'
                  '\n- Useful for survey data with extreme ratings.'
                  '\n- Good when outliers contain important information.'
                  '\n'
                  '\nMinMax Scaler (scales to 0-1 range):'
                  '\n- Scales all values to a fixed range between 0 and 1.'
                  '\n- Good for neural networks that expect bounded inputs.'
                  '\n- Works well with sparse data.'
                  '\n- Preserves zero values in sparse data.')

    if verbose:
        print('\nFinal candidate features for scaling are:')
        print(final_numeric_cols)

    # Process each feature
    for column in final_numeric_cols:
        print()
        if verbose:
            print('-' * 50)
        print(f'Processing feature: "{column}"')
        if verbose:
            print('-' * 50)

        # Check for nulls if warnings aren't suppressed
        null_count = df[column].isnull().sum()
        if null_count > 0 and not skip_warnings:
            print(f'Warning: "{column}" contains {null_count} null values.')
            print('Scaling with null values present may produce unexpected results.')
            if input('Continue scaling this feature? (Y/N): ').lower() != 'y':
                continue

        # Check for infinite values if warnings aren't suppressed
        inf_count = np.isinf(df[column]).sum()
        if inf_count > 0 and not skip_warnings:
            print(f'Warning: "{column}" contains {inf_count} infinite values.')
            print('Scaling with infinite values present may produce unexpected results.')
            if input('Continue scaling this feature? (Y/N): ').lower() != 'y':
                continue

        # Skip constant features
        if df[column].nunique() <= 1:
            print(f'Skipping "{column}" - this feature has no variance.')
            continue

        # Check for extreme skewness if warnings aren't suppressed
        if not skip_warnings:
            skewness = df[column].skew()
            if abs(skewness) > 2:  # Common threshold for "extreme" skewness
                print(f'Warning: "{column}" is highly skewed (skewness={skewness:.2f}).')
                print('Consider applying a transformation before scaling.')
                if input('Continue scaling this feature? (Y/N): ').lower() != 'y':
                    continue

        if verbose:
            # Show current distribution statistics
            print(f'\nCurrent statistics for "{column}":')
            print(df[column].describe())

            # Offer distribution plot
            if input('\nWould you like to see a distribution plot of the feature? (Y/N): ').lower() == 'y':
                try:
                    plt.figure(figsize=(12, 8))
                    sns.histplot(data=df, x=column)
                    plt.title(f'Distribution of {column}')
                    plt.xlabel(column)
                    plt.ylabel('Count')
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                except Exception as exc:
                    print(f'Error creating plot: {exc}')
                    if plt.get_fignums():
                        plt.close('all')

        # Show scaling options
        print('\nSelect scaling method:')
        print('1. Standard Scaler (Z-score normalization)')
        print('2. Robust Scaler (median and IQR based)')
        print('3. MinMax Scaler (0-1 range)')
        print('4. Skip scaling for this feature')

        while True:
            method = input('Enter choice (1, 2, 3, or 4): ')
            # Exit loop if valid input provided
            if method in ['1', '2', '3', '4']:
                break

            # Catch invalid input
            print('Invalid choice. Please enter 1, 2, 3, or 4.')

        # Skip scaling for a given feature if user decides to do so
        if method == '4':
            if verbose:
                print(f'\nSkipping scaling for feature "{column}".')
            continue

        try:
            # Reshape data for scikit-learn
            reshaped_data = df[column].values.reshape(-1, 1)

            # Apply selected scaling method
            if method == '1':
                scaler = StandardScaler()
                method_name = 'Standard'

            elif method == '2':
                scaler = RobustScaler()
                method_name = 'Robust'

            else:
                scaler = MinMaxScaler()
                method_name = 'MinMax'

            # Perform scaling
            df[column] = scaler.fit_transform(reshaped_data)
            scaled_features.append(f'{column} ({method_name})')

            if verbose:
                print(f'\nSuccessfully scaled "{column}" using {method_name} scaler.')

        except Exception as exc:
            print(f'Error scaling feature "{column}": {exc}')
            continue

    # Print summary if features were scaled
    if scaled_features:
        if verbose:
            print('-' * 50)
            print('SCALING SUMMARY:')
            for feature in scaled_features:
                print(f'- {feature}')
            print('-' * 50)

    if verbose:
        print('Scaling complete. Returning modified dataframe.')
        print('-' * 50)

    # Return the modified dataframe with scaled values
    return df


def _prep_df_core(
        df: pd.DataFrame,
        features_to_encode: list[str] | None = None,
        features_to_scale: list[str] | None = None
) -> pd.DataFrame:
    """Core function implementing the TADPREP pipeline with parameter control."""

    def get_bool_param(param_name: str, default: bool = True) -> bool:
        """Get boolean parameter values from user input."""
        while True:
            print(f'\nSet {param_name} parameter:')
            print(f'1. True (default: {default})')
            print('2. False')
            choice = input('Enter choice (1/2) or press Enter to accept default setting: ').strip()

            # Handle the three valid input cases
            if not choice:  # User accepts default
                return default

            elif choice == '1':  # User selects True
                return True

            elif choice == '2':  # User selects False
                return False

            else:  # Invalid input - notify and retry
                print('Invalid choice. Please enter 1, 2, or press Enter.')

    def get_feature_selection(df: pd.DataFrame, operation: str) -> list[str] | None:
        """Get feature selections for encoding and scaling from user input."""
        while True:
            # Present options for feature selection
            print(f'\nFeature Selection for {operation}:')
            print('1. Auto-detect features')
            print('2. Manually select features')
            choice = input('Enter choice (1/2): ').strip()

            if choice == '1':  # Auto-detect
                return None

            elif choice == '2':  # Manual selection
                # Show available features
                print('\nAvailable features:')
                for idx, col in enumerate(df.columns, 1):
                    print(f'{idx}. {col}')

                # Get and validate feature selection
                while True:
                    selections = input(
                        '\nEnter feature numbers (comma-separated) or press Enter to auto-detect: ').strip()

                    if not selections:  # User wants auto-detect
                        return None

                    try:
                        # Convert input to indices then validate
                        indices = [int(idx.strip()) - 1 for idx in selections.split(',')]
                        if not all(0 <= idx < len(df.columns) for idx in indices):
                            print('Error: Invalid feature numbers')
                            continue
                        # Return selected feature names
                        return [df.columns[i] for i in indices]

                    except ValueError:
                        print('Invalid input. Please enter comma-separated numbers.')

            else:  # Invalid choice
                print('Invalid choice. Please enter 1 or 2.')

    # Step 1: File Info
    user_choice = input('\nSTAGE 1: Display file info? (Y/N/Q): ').lower()
    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        _df_info_core(df, verbose=verbose)

    elif user_choice == 'q':
        return df

    # Step 2: Reshape - handles missing values and feature dropping
    user_choice = input('\nSTAGE 2: Run file reshape process? (Y/N/Q): ').lower()

    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        df = _reshape_core(df, verbose=verbose)

    elif user_choice == 'q':
        return df

    # Step 3: Rename and Tag - handles feature renaming and classification
    user_choice = input('\nSTAGE 3: Run feature renaming and tagging process? (Y/N/Q): ').lower()
    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        tag_features = get_bool_param('tag_features', default=False)
        df = _rename_and_tag_core(df, verbose=verbose, tag_features=tag_features)

    elif user_choice == 'q':
        return df

    # Step 4: Feature Stats - calculates and displays feature statistics
    user_choice = input('\nSTAGE 4: Show feature-level information? (Y/N/Q): ').lower()
    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        summary_stats = get_bool_param('summary_stats', default=False)
        _feature_stats_core(df, verbose=verbose, summary_stats=summary_stats)

    elif user_choice == 'q':
        return df

    # Step 5: Impute - handles missing value imputation
    user_choice = input('\nSTAGE 5: Perform imputation? (Y/N/Q): ').lower()
    if user_choice == 'y':
        verbose = get_bool_param('verbose')
        skip_warnings = get_bool_param('skip_warnings', default=False)
        df = _impute_core(df, verbose=verbose, skip_warnings=skip_warnings)

    elif user_choice == 'q':
        return df

    # Step 6: Encode - handles categorical feature encoding
    user_choice = input('\nSTAGE 6: Encode categorical features? (Y/N/Q): ').lower()
    if user_choice == 'y':
        # Use provided features or get interactive selection
        features_to_encode_final = features_to_encode
        if features_to_encode_final is None:
            features_to_encode_final = get_feature_selection(df, 'encoding')
        verbose = get_bool_param('verbose')
        skip_warnings = get_bool_param('skip_warnings', default=False)
        df = _encode_core(
            df,
            features_to_encode=features_to_encode_final,
            verbose=verbose,
            skip_warnings=skip_warnings
        )

    elif user_choice == 'q':
        return df

    # Step 7: Scale - handles numerical feature scaling
    user_choice = input('\nSTAGE 7: Scale numerical features? (Y/N/Q): ').lower()
    if user_choice == 'y':
        # Use provided features or get interactive selection
        features_to_scale_final = features_to_scale
        if features_to_scale_final is None:
            features_to_scale_final = get_feature_selection(df, 'scaling')
        verbose = get_bool_param('verbose')
        skip_warnings = get_bool_param('skip_warnings', default=False)
        df = _scale_core(
            df,
            features_to_scale=features_to_scale_final,
            verbose=verbose,
            skip_warnings=skip_warnings
        )

    elif user_choice == 'q':
        return df

    return df
