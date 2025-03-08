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
Perform row- or column-dependent instance (row) removals, as well as feature (column) removals.

### Parameters:
- `df` Input Pandas dataframe.
- `features_to_reshape` (list[str]) 
- `verbose` (Boolean, default = True) Controls level of detail in output.

### Returns:
- Modified dataframe containing only the subset of instances not removed by user.

### Current State:
- Docstrings updated for all but drop_columns child func.

- ***UX-focused elements factored out of current build
  - A bit hardline on this for now, avoiding additional user input beyond confirmation of operations.
- Core Functionality:
  - Provides user 3 options for reshaping by removal:
    - 1. Row-dependent row removal
      - Provides user with a default missingness threshold
      - Sums, by row, total missing features
      - Encodes, by row, whether those rows' missingness is at/above threshold
    - 2. Column-dependent row removal
      - Displays row missingness by features provided
      - Removes rows with missingness in 'features_to_reshape'
    - 3. Column removal
      - This might not need to exist outside of UX
      - Would likely just be a pandas wrapper

- "Verbosity" differentiation not fully implemented
  - I'm leaning toward this as a UX-side feature of the package
  - Full method functionality is a bit dense with all of the "possible" info the user is getting or not.

### Observed Bugs/Problems:
- testing of current iteration incomplete
  - bugs not detected so far

### Ideas for Development:
- Data types considered "missing" could be subject to user interpretation
  - Consider implementing options for these
- "Architecture" of package as related to this method needs clarifying w/ Don
  - What functionality should go in `transforms.py` vs `tadprep_interactive.py`

### Method History:
- Alpha build by Don Smith
- Beta build by Gabor Horvath (Current State)


## Class: `PlotHandler`
### Core Purpose:
Produce, store, and compare relevant, straightforward visualizations on a per-feature basis as guided by user.

### Methods:
- `.plot_data(self, df, col_name):`
  -  Generates and stores a Seaborn plot for a specified pandas DataFrame column with dtype and "plot type" determined by .det_plot_type().
- `.det_plot_type(self, df, col_name):`
  - Determines an appropriate plot type for a set of data (pd.Series) based on the pandas dtype of the user's DataFrame column.
- `.plot_hist(self, data, col_name):`
  - Draws a Seaborn histogram for numeric-type data and stores a snapshot of the data used to draw it.
- `.plot_box(self, data, col_name):`
  - Draws a Seaborn histogram for categorical-type data and stores a snapshot of the data used to draw it.
- `.plot_line(self, data, col_name):`
  - Draws a Seaborn histogram for timeseries-type data and stores a snapshot of the data used to draw it.
- `.plot_scatter(self, data, col_name):`
  - Draws a Seaborn histogram for mixed-type data and stores a snapshot of the data used to draw it.

### Returns:
- None - is Class

### Current State:
- Alpha build
- Colorblind color palette active
- Data "snapshot" storage system implemented
  - Relies on storing a pd.Series of the data used for a given viz when that viz is created
  - Plot "recall/redraw" and basic comparative viz functionality implemented for histplots
    - `plot_data`, `det_plot_type`, `plot_hist`, `recall_plot`, and `compare_plots` all adjusted
    - plt.subplots() implementation with control-flow for proper axes object positioning and labeling

### Observed Bugs/Problems:
- `__init__.py` and `tadprep` import functionality has issues recognizing the `PlotHandler` class
  - Have to specify the following to enable instantiation of PlotHandler() class objects for testing:
    - 'from tadprep.core.tansforms import PlotHandler'
    - Unsure if this is expected behavior or if is issue with file structure

- `compare_plots` method shows a blank plt.subplots() figure in an unidentified case where 'subplots_nrows' == 1


### Ideas for Development:
- Testing will help indicate whether we should refactor `.det_plot_type()` and `_rename_and_tag_core` for more effective plot type determination.

- As color-blind friendly as we are capable of!
- Static viz only
- Should this be for in-package viz only or produce files for export?
  - File specifications could get tricky

### Method History:
- Alpha build by Gabor Horvath


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
- `summary_stats` (Boolean, default = False) which controls whether summary statistics tables grouped by feature type are displayed.

### Returns:
- None. This is a void method that prints information to the console.

### Current State:
- Core Functionality (Always Run):
  - Separates features into categories by type:
    - Categorical features
    - Numerical features
  - For all features:
    - Shows missingness information (count and percentage)
  - For categorical features:
    - Shows unique value counts
    - Shows mode values
    - Shows category distributions
  - For numerical features:
    - Shows mean
    - Shows range (min/max)
    - Shows basic descriptive statistics
  - Validates all features before processing
  - Ensures data type appropriate statistics


- If `verbose=False`:
  - Shows only feature names and types
  - Shows basic statistics without formatting
  - Shows minimal explanatory text
  - Presents condensed output


- If `verbose=True`, the method **also** provides:
  - Detailed feature type categorization
  - Full descriptive statistics
  - Formatted output with visual separators
  - For categorical features:
    - Complete value distribution information
  - For numerical features:
    - Extended descriptive statistics including median and standard deviation
    

- If `summary_stats=True`, the method **also** provides:
  - Summary tables grouped by feature type
  - Aggregate statistics for each feature class

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- Enhanced Feature Detection:
  - Add identification of boolean features
  - Add identification of datetime features
  - Separate display of statistics by four types (boolean, datetime, categorical, numerical)

- Enhanced Statistical Information:
  - For Numerical Features:
    - Add skewness measurement
    - Add kurtosis measurement
    - Add quartile information
    - Add coefficient of variation
    - Add explanations of these statistics in verbose mode only
  - For Categorical Features:
    - Add entropy/information content
    - Add frequency ratios
    - Add explanations of these metrics in verbose mode only
  - For Datetime Features:
    - Add temporal range statistics
    
- Data Quality Indicators:
  - Add detection of zero-variance features
  - Add detection of near-constant features (>95% single value)
  - Add detection of suspicious patterns
  - Add detection of duplicate features

- Format Improvements:
  - Improve formatting of large numbers
  - Add percentage calculations where relevant
  - Improve floating point number rounding
  - Add clearer section separators

### Method History:
- Alpha build by Don Smith (Current State)


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
