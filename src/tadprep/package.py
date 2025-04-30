import pandas as pd
from .core.transforms import (
    _df_info_core,
    _reshape_core,
    # _find_outliers_core,
    # _find_corrs_core,
    _subset_core,
    _rename_and_tag_core,
    _feature_stats_core,
    _impute_core,
    _encode_core,
    _scale_core,
    _prep_df_core,
    _transform_core,
    _extract_datetime_core
)


def df_info(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Prints comprehensive information about a DataFrame's structure, contents, and potential data quality issues.

    Basic information (always shown):
    - Total number of instances (rows)
    - Total number of features (columns)
    - Count and percentage of instances containing any missing values
    - Count and percentage of duplicate instances, if any exist

    When verbose=True, the method also displays:
    - Features with very low variance (>95% single value)
    - Features containing infinite values (in numeric columns)
    - Features containing empty strings (distinct from NULL/NaN values)
    - Names and data types of all features
    - Non-null value count for each feature

    The method also provides contextual warnings about data quality issues that may require investigation.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be analyzed
    verbose : bool, default=True
        Controls whether detailed feature information and data quality checks are displayed

    Returns
    -------
    None
        This is a void method which prints information to the console.

    Notes
    -----
    Use verbose=True (default) to receive full information about data structure and potential quality issues.
    Use verbose=False when you only need basic information about dataset size and contents.

    Different from NULL/NaN values, empty strings ('') are valid string values that may indicate data quality
    issues in text columns.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': ['x', 'y', 'z', 'w']
    ... })
    >>> tadprep.df_info(df, verbose=True)  # Shows full dataframe information
    >>> tadprep.df_info(df, verbose=False)  # Shows reduced dataframe information
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    _df_info_core(df, verbose)


def reshape(df: pd.DataFrame, features_to_reshape: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Interactively reshapes the input DataFrame according to user specification.

    Allows deletion of missing values, dropping columns, and random sub-setting of instances.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be reshaped
    features_to_reshape : list[str] | None, default=None
        Optional list of features to reshape. If None, the method will default to all features.
    verbose : bool, default=True
        Controls whether detailed process information is displayed

    Returns
    -------
    pandas.DataFrame
        The reshaped DataFrame as modified by the user's specifications

    Examples
    --------
        >>> import pandas as pd
        >>> import tadprep
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6]})
        >>> df_reshaped = tadprep.reshape(df)  # Shows detailed status messages
        >>> df_reshaped_quiet = tadprep.reshape(df, verbose=False)  # Shows only necessary user prompts
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _reshape_core(df, features_to_reshape, verbose)


def find_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = None, verbose: bool = True) -> dict:
    """
    Detects outliers in numerical features of a DataFrame using a specified detection method.

    Analyzes numerical features in the dataframe and identifies outliers using the specified detection
    method. Supports three common approaches for outlier detection: IQR-based detection, Z-score method, and
    Modified Z-score method.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze for outliers
    method : str, default='iqr'
        Outlier detection method to use.
        Options:
          - 'iqr': Interquartile Range (default)
          - 'zscore': Standard Z-score
          - 'modified_zscore': Modified Z-score
    threshold : float, default=None
        Threshold value for outlier detection. If None, uses method-specific defaults:
          - For IQR: 1.5 × IQR
          - For Z-score: 3.0 standard deviations
          - For Modified Z-score: 3.5
    verbose : bool, default=True
        Whether to print detailed information about outliers

    Returns
    -------
    dict
        A dictionary containing outlier information with summary and feature-specific details

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({'A': [1, 2, 3, 100], 'B': ['x', 'y', 'z', 'w']})
    >>> outlier_results = tp.find_outliers(df)  # Use default IQR method
    >>> outlier_results = tp.find_outliers(df, method='zscore')  # Use Z-score method
    >>> outlier_results = tp.find_outliers(df, verbose=False)  # Hide detailed output
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _find_outliers_core(df, method=method, threshold=threshold, verbose=verbose)


def find_corrs(df: pd.DataFrame, method: str = 'pearson', threshold: float = 0.8, verbose: bool = True) -> dict:
    """
    Detects highly-correlated features in a DataFrame using the specified correlation method.

    Analyzes numerical features in the dataframe and identifies feature pairs with correlation
    coefficients exceeding the specified threshold. High correlations often indicate redundant
    features that could be simplified or removed to improve model performance and interpretability.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze for feature correlations
    method : str, default='pearson'
        Correlation method to use.
        Options:
          - 'pearson': Standard correlation coefficient (default, best for linear relationships)
          - 'spearman': Rank correlation (robust to outliers and non-linear relationships)
          - 'kendall': Rank correlation (more robust for small samples, handles ties differently)
    threshold : float, default=0.8
        Absolute correlation coefficient threshold above which features are considered highly correlated.
        Values should be between 0 and 1.
    verbose : bool, default=True
        Whether to print detailed information about detected correlations

    Returns
    -------
    dict
        A dictionary containing correlation information with summary statistics and detailed pair information.
        Structure:
        {
            'summary': {
                'method': str,                # Correlation method used
                'num_correlated_pairs': int,  # Total number of highly correlated pairs
                'max_correlation': float,     # Maximum correlation found
                'avg_correlation': float,     # Average correlation among high pairs
                'features_involved': list,    # List of features involved in high correlations
            },
            'correlation_pairs': [
                {
                    'feature1': str,          # Name of first feature
                    'feature2': str,          # Name of second feature
                    'correlation': float,     # Correlation coefficient
                    'abs_correlation': float  # Absolute correlation value
                },
                ...
            ]
        }

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [5, 4, 3, 2, 1],
    ...     'C': [1, 3, 5, 7, 9],
    ...     'D': [2, 4, 6, 8, 10]
    ... })
    >>> # Use default Pearson correlation
    >>> corr_results = tp.find_corrs(df)
    >>> # Use Spearman correlation with lower threshold
    >>> corr_results = tp.find_corrs(df, method='spearman', threshold=0.6)
    >>> # Hide detailed output
    >>> corr_results = tp.find_corrs(df, verbose=False)
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _find_corrs_core(df, method=method, threshold=threshold, verbose=verbose)


def subset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Interactively subsets the input DataFrame according to user specification. Supports random sampling
    (with or without a seed), stratified sampling, and time-based instance selection for timeseries data.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be reshaped
    verbose : bool, default=True
        Controls whether detailed process information and methodological guidance is displayed

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame as subset by the user's specifications

    Examples
    --------
        >>> import pandas as pd
        >>> import tadprep
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, 5, 6]})
        >>> df_subset = tadprep.subset(df)  # Shows detailed status messages and guidance
        >>> df_subset_quiet = tadprep.subset(df, verbose=False)  # Shows only necessary user prompts
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _subset_core(df, verbose)


def rename_and_tag(df: pd.DataFrame, verbose: bool = True, tag_features: bool = False) -> pd.DataFrame:
    """
    Interactively renames features and allows user to tag them as ordinal or target features, if desired.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose features need to be renamed and/or tagged
    verbose : bool, default = True
        Controls whether detailed process information is displayed
    tag_features : default = False
        Controls whether activate the feature-tagging process is activated

    Returns
    -------
    pandas.DataFrame
        The DataFrame with renamed/tagged features

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({'feature1': [1,2,3], 'feature2': ['a','b','c']})
    >>> df_renamed = tadprep.rename_and_tag(df)  # Only rename features
    >>> df_tagged = tadprep.rename_and_tag(df, tag_features=True)  # Rename and tag features
    >>> df_renamed_quiet = tadprep.rename_and_tag(df, verbose=False, tag_features=False)  # Show minimal output
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _rename_and_tag_core(df, verbose=verbose, tag_features=tag_features)


def feature_stats(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Displays feature-level statistics for each feature in the DataFrame.

    For each feature, displays missingness information and appropriate descriptive statistics
    based on the feature's datatype (boolean, datetime, categorical, or numerical).
    Features are automatically classified by type for appropriate statistical analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze
    verbose : bool, default=True
        Whether to print detailed statistical information and more extensive visual formatting

    Returns
    -------
    None
        This is a void method that prints information to the console.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': ['x', 'y', 'z', 'w']
    ... })
    >>> tadprep.feature_stats(df)  # Show detailed statistics with formatting
    >>> tadprep.feature_stats(df, verbose=False)  # Show only key feature-level statistics
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    _feature_stats_core(df, verbose=verbose)


def impute(df: pd.DataFrame, verbose: bool = True, skip_warnings: bool = False) -> pd.DataFrame:
    """
    Interactively imputes missing values in the DataFrame using simple imputation methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing missing values to impute
    verbose : bool, default = True
        Controls whether detailed process information is displayed
    skip_warnings : bool, default = False
        Controls whether missingness threshold warnings are displayed

    Returns
    -------
    pandas.DataFrame
        The DataFrame with imputed values

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({'A': [1, None, 3], 'B': ['x', 'y', None]})
    >>> df_imputed = tadprep.impute(df)  # Full guidance and warnings
    >>> df_imputed_quiet = tadprep.impute(df, verbose=False)  # Minimize output
    >>> df_imputed_nowarn = tadprep.impute(df, skip_warnings=True)  # Skip missingness warnings
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    return _impute_core(df, verbose=verbose, skip_warnings=skip_warnings)


def encode(
    df: pd.DataFrame,
    features_to_encode: list[str] | None = None,
    verbose: bool = True,
    skip_warnings: bool = False,
    preserve_features: bool = False
) -> pd.DataFrame:
    """
    Interactively encodes categorical features in the DataFrame using specified encoding methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing features to encode.
    features_to_encode : list[str] | None, default=None
        Optional list of features to encode - if None, method will help identify categorical features.
    verbose : bool, default=True
        Controls whether detailed guidance and explanations are displayed.
    skip_warnings : bool, default=False
        Controls whether all best-practice-related warnings about encoding are skipped.
    preserve_features : bool, default=False
        Whether to keep original features in the DataFrame alongside encoded ones.
        When True, original categorical columns are retained after encoding.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with encoded categorical features

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({'A': ['cat', 'dog', 'horse'], 'B': [1, 2, 3]})
    >>> df_encoded = tadprep.encode(df)  # Let function identify categorical features
    >>> df_encoded_specified = tadprep.encode(df, features_to_encode=['A'])  # Specify features to encode
    >>> df_encoded_quiet = tadprep.encode(df, verbose=False)  # Minimize output
    >>> df_encoded_nowarn = tadprep.encode(df, skip_warnings=True)  # Skip best-practice warnings
    >>> df_encoded_preserved = tadprep.encode(df, preserve_features=True)  # Keep original features
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_encode if provided by user
    if features_to_encode is not None:
        if not isinstance(features_to_encode, list):
            raise TypeError('features_to_encode must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_encode):
            raise TypeError('All feature names in features_to_encode must be strings')

        if not all(col in df.columns for col in features_to_encode):
            missing = [col for col in features_to_encode if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate preserve_features parameter
    if not isinstance(preserve_features, bool):
        raise TypeError('preserve_features must be a boolean')

    return _encode_core(
        df,
        features_to_encode=features_to_encode,
        verbose=verbose,
        skip_warnings=skip_warnings,
        preserve_features=preserve_features
    )


def scale(
    df: pd.DataFrame,
    features_to_scale: list[str] | None = None,
    verbose: bool = True,
    skip_warnings: bool = False,
    preserve_features: bool = False
) -> pd.DataFrame:
    """
    Interactively scales numerical features in the DataFrame using standard scaling methods.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing features to scale.
    features_to_scale : list[str] | None, default=None
        Optional list of features to scale - if None, method will help identify numerical features.
    verbose : bool, default=True
        Controls whether detailed guidance and explanations are displayed.
    skip_warnings : bool, default=False
        Controls whether all best-practice-related warnings about scaling are skipped.
    preserve_features : bool, default=False
        Controls whether original features are preserved when scaling. When True, creates new columns
        with the naming pattern '{original_column}_scaled'. If a column with that name already exists,
        a numeric suffix is added: '{original_column}_scaled_1'.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with scaled numerical features. If preserve_features=True, original
        features are retained and new columns are added with scaled values.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    >>> # Basic usage - let function identify numerical features
    >>> df_scaled = tadprep.scale(df)
    >>> # Specify features to scale
    >>> df_scaled_specified = tadprep.scale(df, features_to_scale=['A'])
    >>> # Minimize output
    >>> df_scaled_quiet = tadprep.scale(df, verbose=False)
    >>> # Skip best-practice warnings
    >>> df_scaled_nowarn = tadprep.scale(df, skip_warnings=True)
    >>> # Preserve original features, creating new scaled columns
    >>> df_with_both = tadprep.scale(df, preserve_features=True)
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_scale if provided
    if features_to_scale is not None:
        if not isinstance(features_to_scale, list):
            raise TypeError('features_to_scale must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_scale):
            raise TypeError('All feature names in features_to_scale must be strings')

        if not all(col in df.columns for col in features_to_scale):
            missing = [col for col in features_to_scale if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate preserve_features parameter
    if not isinstance(preserve_features, bool):
        raise TypeError('preserve_features must be a boolean')

    return _scale_core(
        df,
        features_to_scale=features_to_scale,
        verbose=verbose,
        skip_warnings=skip_warnings,
        preserve_features=preserve_features
    )


def prep_df(
        df: pd.DataFrame,
        features_to_encode: list[str] | None = None,
        features_to_scale: list[str] | None = None
) -> pd.DataFrame:
    """
    Run the complete TADPREP pipeline with user control over each step.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to process using the TADPREP pipeline
    features_to_encode : list[str] | None, default=None
        Optional pre-defined list of features to encode. If None, the function facilitates user selection.
    features_to_scale : list[str] | None, default=None
        Optional pre-defined list of features to scale. If None, the function facilitates user selection.

    Returns
    -------
    pandas.DataFrame
        The processed DataFrame after running the specified pipeline steps.

    Notes
    -----
    - At each step, users can:
      - Enter 'Y' to run a given step
      - Enter 'N' to skip a given step
      - Enter 'Q' to quit the pipeline altogether

    - For boolean parameters:
      - Press Enter to accept the default value
      - Enter '1' to set the parameter to True
      - Enter '2' to set the parameter to False

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, None, 4],
    ...     'B': ['x', 'y', 'z', 'w']
    ... })
    >>> processed_df = tadprep.prep_df(df)  # Run pipeline with interactive feature selection
    >>> processed_df = tadprep.prep_df(df, features_to_encode=['B'])  # Run pipeline with pre-defined features
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_encode if provided
    if features_to_encode is not None:
        if not isinstance(features_to_encode, list):
            raise TypeError('The "features_to_encode" object must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_encode):
            raise TypeError('All feature names in features_to_encode must be strings')

        if not all(col in df.columns for col in features_to_encode):
            missing = [col for col in features_to_encode if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate features_to_scale if provided
    if features_to_scale is not None:
        if not isinstance(features_to_scale, list):
            raise TypeError('The "features_to_scale" object must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_scale):
            raise TypeError('All feature names in features_to_scale must be strings')

        if not all(col in df.columns for col in features_to_scale):
            missing = [col for col in features_to_scale if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    print('Welcome to the TADPREP pipeline.')
    print('At each step you can:')
    print("- Enter 'Y' to execute the step")
    print("- Enter 'N' to skip the step")
    print("- Enter 'Q' to quit the pipeline")
    print('\nFor boolean parameters:')
    print('- Press Enter to accept the default value')
    print("- Enter '1' to set the parameter to True")
    print("- Enter '2' to set the parameter to False")

    return _prep_df_core(df, features_to_encode=features_to_encode, features_to_scale=features_to_scale)


def transform(
        df: pd.DataFrame,
        features_to_transform: list[str] | None = None,
        verbose: bool = True,
        preserve_features: bool = False,
        skip_warnings: bool = False
) -> pd.DataFrame:
    """
    Interactively transforms numerical features using various mathematical transformations.

    Applies transformations to improve data distributions for modeling, with a focus on
    normalization and linearization. The function analyzes data characteristics and
    suggests appropriate transformations based on distribution properties.

    Supports various transformations including:
    - Logarithmic: log, log10, log1p (for handling different data requirements)
    - Power: sqrt, square, cube, reciprocal
    - Statistical: Box-Cox, Yeo-Johnson (for distribution normalization)

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing features to transform.
    features_to_transform : list[str] | None, default=None
        Optional list of features to transform. If None, method will help identify
        numerical features and exclude likely categorical ones.
    verbose : bool, default=True
        Controls whether detailed guidance, explanations, and visualizations are displayed.
        When True, offers distribution plots and transformation explanations.
    preserve_features : bool, default=False
        Controls whether original features are preserved when transforming. When True,
        creates new columns with the naming pattern '{original_column}_transformed'.
        If a column with that name already exists, a numeric suffix is added:
        '{original_column}_transformed_1'.
    skip_warnings : bool, default=False
        Controls whether all best-practice-related warnings about distributions and
        nulls are skipped. Setting to True streamlines the process for experienced users.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with transformed features. If preserve_features=True, original
        features are retained and new columns are added with transformed values.

    Notes
    -----
    Some transformations have specific data requirements:
    - Log and Box-Cox require strictly positive values (no zeros or negatives)
    - Log1p and sqrt require non-negative values (no negatives)
    - Reciprocal cannot handle zero values

    The function automatically identifies which transformations are valid for each feature
    based on its characteristics (presence of zeros, negative values, etc.).

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({'A': [1, 2, 100, 200], 'B': ['x', 'y', 'z', 'w']})
    >>> # Basic usage - let function identify numerical features
    >>> df_transformed = tp.transform(df)
    >>> # Specify features to transform
    >>> df_transformed_specified = tp.transform(df, features_to_transform=['A'])
    >>> # Minimize output
    >>> df_transformed_quiet = tp.transform(df, verbose=False)
    >>> # Skip distribution warnings
    >>> df_transformed_nowarn = tp.transform(df, skip_warnings=True)
    >>> # Preserve original features, creating new transformed columns
    >>> df_with_both = tp.transform(df, preserve_features=True)
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate features_to_transform if provided
    if features_to_transform is not None:
        if not isinstance(features_to_transform, list):
            raise TypeError('features_to_transform must be a list of strings')

        if not all(isinstance(col, str) for col in features_to_transform):
            raise TypeError('All feature names in features_to_transform must be strings')

        if not all(col in df.columns for col in features_to_transform):
            missing = [col for col in features_to_transform if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate preserve_features parameter
    if not isinstance(preserve_features, bool):
        raise TypeError('preserve_features must be a boolean')

    # Validate skip_warnings parameter
    if not isinstance(skip_warnings, bool):
        raise TypeError('skip_warnings must be a boolean')

    # Call the core implementation
    return _transform_core(
        df,
        features_to_transform=features_to_transform,
        verbose=verbose,
        preserve_features=preserve_features,
        skip_warnings=skip_warnings
    )


def extract_datetime(
        df: pd.DataFrame,
        datetime_features: list[str] | None = None,
        verbose: bool = True,
        preserve_features: bool = False,
        components: list[str] | None = None
) -> pd.DataFrame:
    """
    Extracts useful features from datetime columns in a dataframe, converting temporal
    information into features that machine learning models can use more effectively.

    This function identifies datetime columns, extracts selected components (year, month, day, etc.),
    and creates new features. It can also detect time series data and offer additional
    time-specific components.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing datetime features to extract.
    datetime_features : list[str] | None, default=None
        Optional list of datetime features to process. If None, the function will
        identify datetime features interactively.
    verbose : bool, default=True
        Controls whether detailed guidance and explanations are displayed.
    preserve_features : bool, default=False
        Controls whether original datetime features are preserved in the DataFrame.
        When False, original datetime columns are removed after extraction.
    components : list[str] | None, default=None
        Optional list of specific datetime components to extract. If None, the function
        will help identify components interactively.

        Standard components include:
        - year, month, day, quarter
        - dayofweek, weekofyear, dayofyear
        - hour, minute

        For time series data, additional components include:
        - is_weekend, is_month_start, is_month_end
        - is_quarter_start, is_quarter_end

    Returns
    -------
    pandas.DataFrame
        Modified dataframe with extracted datetime features. If preserve_features=True,
        original datetime columns are retained.

    Examples
    --------
    >>> import pandas as pd
    >>> import tadprep as tp
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2021-01-01', periods=5),
    ...     'value': [10, 20, 30, 40, 50]
    ... })
    >>> # Basic usage - let function identify components interactively
    >>> df_extracted = tp.extract_datetime(df)
    >>> # Specify datetime columns and components
    >>> df_extracted = tp.extract_datetime(
    ...     df,
    ...     datetime_features=['date'],
    ...     components=['year', 'month', 'dayofweek']
    ... )
    >>> # Keep original datetime columns
    >>> df_extracted = tp.extract_datetime(df, preserve_features=True)
    >>> # Minimize output
    >>> df_extracted = tp.extract_datetime(df, verbose=False)
    """
    # Ensure input is a Pandas dataframe
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')

    # Ensure dataframe is not empty
    if df.empty:
        raise ValueError('Input DataFrame is empty')

    # Validate datetime_features if provided
    if datetime_features is not None:
        if not isinstance(datetime_features, list):
            raise TypeError('datetime_features must be a list of strings')

        if not all(isinstance(col, str) for col in datetime_features):
            raise TypeError('All feature names in datetime_features must be strings')

        if not all(col in df.columns for col in datetime_features):
            missing = [col for col in datetime_features if col not in df.columns]
            raise ValueError(f'Features not found in DataFrame: {missing}')

    # Validate components if provided
    valid_components = [
        'year', 'month', 'day', 'dayofweek', 'hour', 'minute',
        'second', 'quarter', 'weekofyear', 'dayofyear',
        'is_weekend', 'is_month_start', 'is_month_end',
        'is_quarter_start', 'is_quarter_end'
    ]

    if components is not None:
        if not isinstance(components, list):
            raise TypeError('components must be a list of strings')

        if not all(isinstance(comp, str) for comp in components):
            raise TypeError('All component names must be strings')

        invalid = [comp for comp in components if comp not in valid_components]
        if invalid:
            raise ValueError(f'Invalid component names: {invalid}. Valid components are: {valid_components}')

    # Validate preserve_features
    if not isinstance(preserve_features, bool):
        raise TypeError('preserve_features must be a boolean')

    # Call core implementation
    return _extract_datetime_core(
        df,
        datetime_features=datetime_features,
        verbose=verbose,
        preserve_features=preserve_features,
        components=components)
