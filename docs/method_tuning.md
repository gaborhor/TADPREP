# Public-Facing Method Development Planning

## Method: `df_info`
### Core Purpose:
Prints summary, top-level information about a dataframe to the console.

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail in output.

### Returns:
- None - void method. Prints info to console.

### Current State:
- If `verbose=False`:
  - Prints number of instances (rows) in df
  - Prints number of features (columns) in df
  - Prints total number of instances with *any* missing values
  - Prints % of instances with *any* missing values as a proportion of total instances
  - Prints count of duplicate instances


- If `verbose=True`, the method **also** prints:
  - A warning to assess whether the existence of duplicate instances indicates a data error has occurred
  - A list of near-constant features (i.e. those with >= 95% matched values)
  - A list of features containing any infinite values (i.e. np.inf values)
  - A list of object-type features containing *empty* strings (i.e. distinct from NULL/NaN values)
  - Names and datatypes of all features, along with memory use, etc. *(This is a call to Pandas' .info() method)*
  - A line of dashes on either side of the .info() output for visual separation

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History
 - Alpha build by Don Smith
 - Beta build by Don Smith (Current State)


## Method: `reshape`
### Core Purpose:

### Parameters:

### Returns:

### Current State:

### Observed Bugs/Problems:

### Ideas for Development:

### Method History:
- Alpha build by Don Smith (Current State)


## Method: `subset`
### Core Purpose:
Allows user to subset large datasets using appropriate/desired subset processes.

### Parameters:
- `df` Input Pandas dataframe
- `verbose` (Boolean, default = True) which controls level of detail/guidance in output

### Returns:
- Modified dataframe with subsetted instances as specified by user.

### Current State:

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History:
- Alpha build by Don Smith


## Method: `impute`
### Core Purpose:
Works interactively with user to impute missing values in features using common imputation methods.

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail/guidance in output.
- `skip_warnings` (Boolean, default = False) which controls whether to skip data quality and missingness warnings.

### Returns:
- Modified dataframe with imputed values as specified by user.

### Current State:
- Core Functionality (Always Run):
  - Detects and validates datetime features to identify time series data
  - Checks for "false numeric" features (e.g., 1/0 representations of Yes/No)
  - Performs data quality checks including:
    - Near-zero variance features
    - Features with high outlier counts
    - Highly correlated feature pairs
    - Features with extreme skewness
  - Supports multiple imputation methods:
    - Statistical methods (mean, median, mode)
    - Constant value imputation
    - Random sampling from non-null values
    - Forward/backward fill (for time series data)
  - Tracks imputation actions for summary reporting


- If `verbose=False`:
  - Shows minimal feature type classification
  - Displays only essential user prompts
  - Presents basic imputation choices
  - Shows only critical warnings
  - Provides basic confirmation of successful operations


- If `verbose=True`, additionally provides:
  - Detailed initial and final feature classifications
  - Comprehensive missingness statistics
  - Detailed explanations of imputation methods
  - Pre-imputation feature distributions and statistics
  - Visual distribution plots for numerical features
  - Step-by-step guidance through the imputation process
  - Post-imputation distribution comparisons
  - Comprehensive imputation summary including:
    - Feature names
    - Number of values imputed
    - Methods used
    - Imputation values or approaches


- If `skip_warnings=False`, additionally checks and warns about:
  - Features with high missingness rates (>10%)
  - Data quality issues including:
    - Near-zero variance
    - High outlier counts
    - High correlations between features
    - Extreme skewness
  - Provides detailed guidance on handling problematic features
  - Allows user to 'proceed with caution' after each warning

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History:
- Alpha build by Don Smith
- Beta build by Don Smith (Current State)
