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
Provides interactive functionality for subsetting data through random sampling (with or without seed), 
stratified sampling, or time-based instance selection for timeseries data.

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail/guidance in output.

### Returns:
- Modified dataframe containing only the subset of instances specified by user.

### Current State:
- Core Functionality (Always Run):
  - Detects categorical features for stratified sampling
  - Identifies datetime features and indices for time-based subsetting
  - Automatically converts string datetime columns to datetime type
  - Supports multiple subsetting methods:
    - True random sampling (unseeded)
    - Reproducible random sampling (seeded)
    - Stratified random sampling (if categorical features are present)
    - Time-based subsetting (if datetime elements are present)
  - Validates all user inputs and subset parameters
  - Ensures minimum instance counts are maintained
  - Preserves data integrity during subsetting


- If `verbose=False`:
  - Shows only available subsetting methods
  - Displays minimal progress information
  - Presents only essential user prompts
  - Shows basic confirmation of successful operations
  - Reports final instance count


- If `verbose=True`, the method **also** provides:
  - Detailed explanation of available subsetting methods
  - Comprehensive feature type identification
  - For time series data:
    - Data frequency analysis
    - Time span information
    - Example timestamps
    - Guidance on date format inputs
  - For stratified sampling:
    - Detailed explanation of the process
    - Category distribution information
    - Guidance on maintaining representativeness
  - Detailed summary of subsetting results including:
    - Original instance count
    - Subset instance count
    - Subsetting method used
    - Additional method-specific details:
      - Time boundaries (for time-based subsetting)
      - Category proportions (for stratified sampling)

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- Possible additional subset/sampling methodologies:
  - Option for weighted random sampling
  - Option for bootstrap sampling

### Method History:
- Alpha build by Don Smith (Current State)


## Method: `rename_and_tag`
### Core Purpose:
Provides interactive functionality for renaming features and optionally tagging them as ordinal or target features 
by appending appropriate suffixes ('_ord' or '_target').

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail/guidance in output.
- `tag_features` (Boolean, default = False) which controls whether the feature-tagging process runs.

### Returns:
- Modified dataframe with renamed and/or tagged features.

### Current State:
- Core Functionality (Always Run):
  - Displays enumerated list of current features
  - Validates all feature names and indices
  - Ensures new feature names don't conflict with existing names
  - Provides cancel-out options at each stage
  - Maintains data integrity during renaming
  - Performs extensive feature name validation:
    - Checks for valid Python identifiers
    - Identifies problematic characters
    - Helps avoid undesired anti-patterns
    - Validates against Python keywords
    - Prevents duplicate feature names
  - Tracks all renaming operations for reporting
  - Provides operation cancellation options at each stage


- Parameter `tag_features` Controls Tagging Process Activation:
  - When `tag_features=False` (Default):
    - Only feature renaming functionality is available
    - Feature tagging stages are completely skipped
    - Process completes after renaming stage


  - When `tag_features=True`:
    - Enables complete feature tagging workflow
    - Adds ordinal feature tagging stage ('_ord' suffix)
    - Adds target feature tagging stage ('_target' suffix)
    - Validates that features aren't already tagged
    - Prevents duplicate tagging of features
    - All stages (rename, ordinal tag, target tag) can be skipped individually


- If `verbose=False`:
  - Shows only basic feature list
  - Displays minimal progress information
  - Presents only essential user prompts
  - Shows basic confirmation of successful operations


- If `verbose=True` (Default), the method **also** provides:
  - Clear stage demarcation with visual separators
  - Process initiation notifications
  - Detailed confirmation of each rename operation
  - Warning messages for problematic feature names
  - Preview and confirmation of each name change
  - For feature tagging:
    - Explanatory messages about tagging process
    - Clear separation between ordinal and target tagging stages
  - Comprehensive summary of all changes made including:
    - Detailed summary table of all operations
    - Changes grouped by operation type
    - Feature renames with before/after values
    - List of tagged features by tag type
    - Clear notation of process completion

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- **Foundational Question:** Is the tagging functionality useful? Is it out of scope for a method that's mostly about 
feature renaming? What useful data science purpose does it serve?

### Method History:
- Alpha build by Don Smith
- Beta build by Don Smith (Current State)


## Method: `feature_stats`

### Core Purpose:
Displays feature-level statistics and information for each feature in a dataframe, categorizing features by type 
and providing appropriate descriptive statistics based on feature type.

### Parameters:
- `df` Input Pandas dataframe.
- `verbose` (Boolean, default = True) which controls level of detail in output.

### Returns:
- None. This is a void method that prints information to the console.

### Current State:
- Core Functionality (Always Run):
  - Separates features into categories by type:
    - Boolean features (logical values or 0/1 integers)
    - Datetime features
    - Categorical features
    - Numerical features
  - For all features:
    - Shows missingness information (count and percentage)
  - For boolean features:
    - Shows true/false value counts and percentages
  - For datetime features:
    - Shows date range information
  - For categorical features:
    - Shows unique value counts
    - Shows mode values
    - Shows category distributions
  - For numerical features:
    - Shows mean
    - Shows range (min/max)
    - Shows basic descriptive statistics
  - Performs data quality checks:
    - Zero-variance features
    - Near-constant features (>95% single value)
    - Potential duplicate features


- If `verbose=False`:
  - Shows only feature names and types
  - Shows basic statistics without formatting
  - Shows minimal explanatory text
  - Presents condensed output


- If `verbose=True`, the method **also** provides:
  - Detailed feature type categorization and distribution
  - Data quality alerts for potential issues
  - Extended statistics:
    - For categorical: entropy values, top frequency ratios, distribution patterns
    - For numerical: quartile information, skewness, kurtosis, coefficient of variation
  - Formatted output with visual separators
  - Contextual interpretation of statistical measures

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- None as of current state

### Method History:
- Alpha build by Don Smith
- Beta build by Don Smith (Current State)


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
