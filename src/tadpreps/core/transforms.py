def _info_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function prints and logs use information about the full, unaltered datafile for the user.
    Args:
        df (pd.DataFrame): A Pandas dataframe containing the full, unaltered dataset.
    Returns:
        None. This is a void function.
    """
    # Print number of instances in file
    print(f'The unaltered file has {df.shape[0]} instances.')  # [0] is rows

    # Print number of features in file
    print(f'The unaltered file has {df.shape[1]} features.')  # [1] is columns
    print('-' * 50)  # Visual separator

    # Print names and datatypes of features in file
    print('Names and datatypes of features:')
    print(df.info(memory_usage=False, show_counts=False))
    print('-' * 50)  # Visual separator

    # Instances with missing values
    row_missing_cnt = df.isnull().any(axis=1).sum()  # Compute count
    row_missing_rate = (row_missing_cnt / len(df) * 100).round(2)  # Compute rate
    print(f'{row_missing_cnt} instances ({row_missing_rate}%) contain at least one missing value.')


def _reshape_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function allows the user to delete instances with any missing values, to drop features from the
    dataset, and to sub-set the data by deleting a specified proportion of the instances at random.
    Args:
        df (pd.DataFrame): The original, unaltered dataset.
    Returns:
        df (pd.DataFrame): The dataset after trimming/sub-setting.
    """
    row_missing_cnt = df.isnull().any(axis=1).sum()  # Compute count
    # Ask if the user wants to delete *all* instances with any missing values, if any exist
    if row_missing_cnt > 0:
        user_drop_na = input('Do you want to drop all instances with *any* missing values? (Y/N): ')
        if user_drop_na.lower() == 'y':
            df = df.dropna()
            print(f'After deletion of instances with missing values, {len(df)} instances remain.')

    # Ask if the user wants to drop any of the columns/features in the dataset
    user_drop_cols = input('Do you want to drop any of the features in the dataset? (Y/N): ')
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
                print(f'Dropped features: {",".join(drop_cols_names)}')  # Log dropped columns
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
                    print('Random sub-setting cancelled.')
                    break

                subset_rate = float(subset_input)  # Convert string input to float
                if 0 < subset_rate < 1:  # If the float is valid (i.e. between 0 and 1)
                    retain_rate = 1 - subset_rate  # Compute retention rate
                    retain_row_cnt = int(len(df) * retain_rate)  # Select count of rows to keep in subset

                    df = df.sample(n=retain_row_cnt)  # No random state set b/c we want true randomness
                    print(f'Randomly dropped {subset_rate}% of instances. {retain_row_cnt} instances remain.')
                    break

                # Catch user input error for invalid/out-of-range float
                else:
                    print('Enter a value between 0.0 and 1.0.')

            # Catch outer-level user input errors
            except ValueError:
                print('Invalid input. Enter a float value between 0.0 and 1.0 or enter "C" to cancel.')
                continue  # Restart the loop

    return df  # Return the trimmed dataframe


def _rename_and_tag_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function allows the user to rename features and to append the '_ord' and/or '_target' suffixes
    to ordinal or target columns.
    Args:
        df (pd.DataFrame): The 'trimmed' dataset created by trim_file().
    Returns:
        df (pd.DataFrame): The dataset with renamed columns.
    """
    # Ask if user wants to rename any columns
    user_rename_cols = input('Do you want to rename any of the features in the dataset? (Y/N): ')
    if user_rename_cols.lower() == 'y':
        print('The list of features currently present in the dataset is:')
        for col_idx, column in enumerate(df.columns, 1):  # Create enumerated list of features starting at 1
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                rename_cols_input = input('\nEnter the index integer of the feature you wish to rename '
                                          'or enter "C" to cancel: ')

                # Check for user cancellation
                if rename_cols_input.lower() == 'c':
                    print('Feature renaming cancelled.')
                    break

                col_idx = int(rename_cols_input)  # Convert input to integer
                if not 1 <= col_idx <= len(df.columns):  # Validate entry
                    raise ValueError('Column index is out of range.')

                # Get new column name from user
                col_name_old = df.columns[col_idx - 1]
                col_name_new = input(f'Enter new name for feature "{col_name_old}": ').strip()

                # Validate name to make sure it doesn't already exist
                if col_name_new in df.columns:
                    print(f'Feature name "{col_name_new}" already exists. Choose a different name.')
                    continue  # Restart the loop

                # Rename column in-place
                df = df.rename(columns={col_name_old: col_name_new})
                print(f'Renamed feature "{col_name_old}" to "{col_name_new}".')

                # Ask if user wants to rename another column
                if input('Do you want to rename another feature? (Y/N): ').lower() != 'y':
                    break

            # Catch input errors
            except ValueError as exc:
                print(f'Invalid input: {exc}')

    # Ask if user wants to perform batch-level appending of '_ord' tag to ordinal features
    user_tag_ord = input('Do you want to tag any features as ordinal by appending the "_ord" suffix '
                         'to their names? (Y/N): ')

    if user_tag_ord.lower() == 'y':
        print('\nCurrent feature list:')
        for col_idx, column in enumerate(df.columns, 1):
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                ord_input = input('\nEnter the index integers of ordinal features (comma-separated) '
                                  'or enter "C" to cancel: ')

                if ord_input.lower() == 'c':  # If user cancels
                    print('Ordinal feature tagging cancelled.')  # Note the cancellation
                    break  # Exit the loop

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
                    break

                df.rename(columns=ord_rename_map, inplace=True)  # Perform tagging
                print(f'Tagged the following features as ordinal: {", ".join(ord_rename_map.keys())}')
                break

            # Catch invalid input
            except ValueError as exc:
                print(f'Invalid input: {exc}')
                continue

    # Ask if user wants to perform batch-level appending of '_target' tag to target features
    user_tag_target = input('Do you want to tag any features as targets by appending the "_target" suffix '
                            'to their names? (Y/N): ')

    if user_tag_target.lower() == 'y':
        print('\nCurrent features:')
        for col_idx, column in enumerate(df.columns, 1):
            print(f'{col_idx}. {column}')

        while True:  # We can justify 'while True' because we have a cancel-out input option
            try:
                target_input = input('\nEnter the index integers of target features (comma-separated) '
                                     'or enter "C" to cancel: ')

                # Check for user cancellation
                if target_input.lower() == 'c':
                    print('Target feature tagging cancelled.')
                    break

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
                    break

                df = df.rename(columns=target_rename_map)  # Perform tagging
                print(f'Tagged the following features as targets: {", ".join(target_rename_map.keys())}')
                break

            # Catch invalid input
            except ValueError as exc:
                print(f'Invalid input: {exc}')
                continue  # Restart the loop

    return df  # Return dataframe with renamed and tagged columns


def _feature_stats_core(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    This function aggregates the features by class (i.e. feature type) and prints top-level missingness and
    descriptive statistics information at the feature level. It then builds and logs summary tables at the feature-class
    level.
    Args:
        df (pd.DataFrame): The renamed/tagged dataframe created by rename_features().
    Returns:
        A tuple of lists of strings for the non-target features at the feature-class level. These are used by the
        scaling and encoding functions.
    """
    print('Displaying top-level information for features in dataset...')
    print('-' * 50)  # Visual separator

    # Create a list of columns which are categorical and do NOT have the '_ord' or '_target' suffixes
    cat_cols = [column for column in df.columns
                if df[column].dtype == 'object'
                and not column.endswith('_ord')
                and not column.endswith('_target')
                ]

    # Create a list of ordinal columns (i.e. having the '_ord') suffix if any are present
    # NOTE: The ordinal columns must NOT have the '_target' suffix - they must not be target features
    ord_cols = [column for column in df.columns
                if column.endswith('_ord')
                and not column.endswith('_target')
                ]

    # Create a list of columns which are numerical, not ordinal, and do NOT have the '_target' suffix
    num_cols = [column for column in df.columns
                if pd.api.types.is_numeric_dtype(df[column])  # Use pandas' built-in numeric type checking
                and not column.endswith('_target')
                and not column.endswith('_ord')]

    # Create a list of target columns (columns with '_target' suffix)
    target_cols = [column for column in df.columns if column.endswith('_target')]

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

                print(f'\nFeature "{column}" has only {len(unique_vals)} unique integer values: {unique_vals}')
                print('ALERT: This could be a categorical feature encoded as numbers, e.g. a 1/0 representation of '
                      'Yes/No values.')

                user_cat = input(f'Should "{column}" actually be treated as categorical? (Y/N): ')  # Ask user to choose
                if user_cat.lower() == 'y':
                    df[column] = df[column].astype(str)  # Cast the values to strings in-place
                    true_cat_cols.append(column)  # Add the identified feature to the true_cat_cols list
                    print(f'Converted numerical feature "{column}" to categorical type.')  # Log the choice
                    print('-' * 50)  # Visual separator

                # Otherwise, if user says no, treat the feature as truly numeric
                else:
                    true_num_cols.append(column)

            # If the feature fails the checks, treat the feature as truly numeric
            else:
                true_num_cols.append(column)

        return true_num_cols, true_cat_cols  # Return the lists of true numerical and identified categorical features

    # Call the helper function
    num_cols, new_cat_cols = handle_numeric_cats(df, num_cols)
    cat_cols.extend(new_cat_cols)  # Add any new identified categorical features to cat_cols

    # Print a notification of whether there are any ordinal-tagged features in the dataset
    if ord_cols:
        print(f'NOTE: {len(ord_cols)} ordinal features are present in the dataset.')
        print('-' * 50)  # Visual separator
    else:
        print('NOTE: No ordinal features are tagged in the dataset.')
        print('-' * 50)  # Visual separator

    # Print the names of the categorical features
    if cat_cols:
        print('The categorical non-target features are:')
        print(', '.join(cat_cols))
        print('-' * 50)  # Visual separator
    else:
        print('No categorical non-target features were found in the dataset.')
        print('-' * 50)  # Visual separator

    # Print the names of the ordinal features (if present)
    if ord_cols:
        print('The ordinal non-target features are:')
        print(', '.join(ord_cols))
        print('-' * 50)  # Visual separator

    # Print the names of the numerical features ('The numerical non-target features are:')
    if num_cols:
        print('The numerical non-target features are:')
        print(', '.join(num_cols))
        print('-' * 50)  # Visual separator
    else:
        print('No numerical non-target features were found in the dataset.')
        print('-' * 50)  # Visual separator

    print('Producing key values at the feature level...')
    print('NOTE: Key values at the feature level are printed but not logged.')  # Notify user

    def show_key_vals(column: str, df: pd.DataFrame, feature_type: str):
        """This helper function calculates and prints key values and missingness info at the feature level."""
        print('-' * 50)  # Visual separator
        print(f'Key values for {feature_type} feature "{column}":')
        print('-' * 50)  # Visual separator

        # Calculate missingness at feature level
        missing_cnt = df[column].isnull().sum()  # Total count
        missing_rate = (missing_cnt / len(df) * 100).round(2)  # Missingness rate
        print(f'Missingness information for "{column}":')
        print(f'{missing_cnt} missing values - ({missing_rate}% missing)')

        # Ensure the feature is not fully null before producing value counts
        if not df[column].isnull().all():
            if feature_type in ['Categorical', 'Ordinal']:  # Note: these are generated and passed in the outer function
                print(f'\nValue counts:')
                print(df[column].value_counts())  # Print value counts
                # Print mode value if present - if multiple modes exist we produce the first mode
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
            show_key_vals(column, df, 'Categorical')

        # Move to feature-class level information
        print('-' * 50)  # Visual separator
        print('Producing aggregated key values for categorical features...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')
        print('-' * 50)  # Visual separator

        # Build and log summary table for categorical features
        cat_summary = pd.DataFrame({
            'Unique_values': [df[column].nunique() for column in cat_cols],
            'Missing_count': [df[column].isnull().sum() for column in cat_cols],
            'Missing_rate': [(df[column].isnull().sum() / len(df) * 100).round(2)
                             for column in cat_cols]
        }, index=cat_cols)
        print('Categorical Features Summary Table:')
        print(str(cat_summary))

    if ord_cols:  # If ordinal features are present
        print('-' * 50)  # Visual separator
        print('KEY VALUES FOR ORDINAL FEATURES:')
        for column in ord_cols:
            show_key_vals(column, df, 'Ordinal')

        # Move to feature-class level information and log
        print('-' * 50)  # Visual separator
        print('Producing aggregated key values for ordinal features...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')
        print('-' * 50)  # Visual separator

        # Build and log summary table for ordinal features
        ord_summary = pd.DataFrame({
            'Unique_values': [df[column].nunique() for column in ord_cols],
            'Missing_count': [df[column].isnull().sum() for column in ord_cols],
            'Missing_rate': [(df[column].isnull().sum() / len(df) * 100).round(2)
                             for column in ord_cols]
        }, index=ord_cols)
        print('Ordinal Features Summary Table:')
        print(str(ord_summary))

    if num_cols:  # If numerical features are present
        print('-' * 50)  # Visual separator
        print('KEY VALUES FOR NUMERICAL FEATURES:')
        for column in num_cols:
            show_key_vals(column, df, 'Numerical')

        # Move to feature-class level information and log
        print('-' * 50)  # Visual separator
        print('Producing aggregated key values for numerical features...')
        print('NOTE: Key values at the feature-class level are both printed and logged.')
        print('-' * 50)  # Visual separator

        # Build and log summary table for numerical features
        num_summary = df[num_cols].describe()
        print('Numerical Features Summary Table:')
        print(str(num_summary))

    # Print key values for target features
    if target_cols:  # If target features are present
        print('-' * 50)  # Visual separator
        print('TARGET FEATURE STATISTICS')
        for column in target_cols:
            # Note that we use the pandas type-assessment method to choose the string to pass to show_key_vals
            show_key_vals(column, df,
                          'Numerical' if pd.api.types.is_numeric_dtype(df[column]) else 'Categorical')

        # Build and log summary table for target features
        # We call .describe() for numerical target features and produce value counts otherwise
        print('-' * 50)  # Visual separator
        print('Target Features Summary Table:')
        for column in target_cols:
            if pd.api.types.is_numeric_dtype(df[column]):
                print(f'Summary statistics for target feature {column}:')
                print(df[column].describe())

            else:
                print(f'Value counts for target feature {column}:')
                print(df[column].value_counts())

    # Return the tuple of the lists of columns by type for use in the encoding and scaling functions
    return cat_cols, ord_cols, num_cols

# TODO: Bring in impute and encode/scale functions
