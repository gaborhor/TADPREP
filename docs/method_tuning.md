# Public-Facing Method Development Planning

## Method: df_info
### Core Purpose:
Prints summary, top-level information about a dataframe to the console.

### Parameters:
- `df` Input Pandas dataframe
- `verbose` (Boolean, default = True) which controls level of detail in output

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


## Method: impute
### Core Purpose:
Works interactively with user to impute missing values in features using common imputation methods.

### Parameters:
- `df` Input Pandas dataframe
- `verbose` (Boolean, default = True) which controls level of detail/guidance in output
- `skip_warnings` (Boolean, default = False) which controls whether to skip missingness threshold warnings

### Returns:
- Modified dataframe with imputed values as specified by user

### Current State:
- If `verbose=False`:
  - Checks for any "false numeric" features, i.e. a categorical feature encoded as numbers, e.g. a 1/0 representation 
of 'Yes/No' values
  - Asks user whether such features (if any are identified) should actually be treated as categorical
  - Asks user whether to impute only for recommended features based on missingness rate or whether to consider *all* 
features as candidates for imputation
  - Asks user to select an appropriate imputation method for each feature, or to skip imputation for that feature


- If `verbose=True`, the method **also** prints:
  - A notification that the imputation process has begun
  - An initial classification of features into type-based "buckets" (i.e. a list of categorical and numerical features 
as defined by their Pandas datatype)
  - A final classification of features into type-based "buckets" *after* the checks for "false numeric" features have 
been run and handled by the user
  - A count and rate of missingness for each feature, along with warnings about bias introduction for sparse features
  - A list of features which are good candidates for imputation based on missingness rates
  - A warning that imputation should be abandoned if no features are assessed as good candidates for imputation
  - An offer to display a brief explanation of supported imputation methods and best practices for the user
  - A message for each imputed feature describing which imputation method and imputed value are being applied
  - A notification that the imputation process is complete


- If `skip_warnings=True`:
  - The warnings about missingness rates and bias introduction are not displayed
  - The list of features considered good imputation candidates is not displayed
  - Any warnings that imputation is not appropriate for the given dataset are not displayed

### Observed Bugs/Problems:
- None as of current state

### Ideas for Development:
- DATA QUALITY CHECKS:
  - Add checks for:
    - Zero-variance features
    - Highly correlated features
    - Extreme outliers
  - These checks will run at the same time as the current missingness analysis
  
    - When skip_warnings=False:
        - Modify the "good candidates" list based on these checks
        - Warn about potentially problematic features

    - When verbose=True:
      - Provide detailed information about detected issues
      - Explain why these characteristics matter for imputation
      - Show specific values/thresholds that triggered warnings


- ADDITIONAL IMPUTATION METHODS:
  - Forward/Backward Fill *(Only offered for time series data - Need to search for datetime features to assess and 
ask user if the data are timeseries in form)*

    - When verbose=True:
      - Explain when these methods are appropriate
      - Explain difference between forward/backward fill
      - Show examples of how the values would be propagated


  - Constant Value Imputation (Available for all feature types)
    - Examples:
      - Using 0 for missing values in count data
      - Using -1 for missing categorical codes
      - Using domain-specific default values
  
      - When verbose=True:
        - Explain common use cases
        - Provide examples of appropriate constant values
        - Warn about potential impacts on distribution


  - Random Sampling (Available for all feature types)

      - When verbose=True:
        - Explain how it can preserve feature distribution
        - Compare/contrast with other imputation methods


- MORE USER FEEDBACK:
  - When verbose=True:

    - Pre/Post Imputation Comparisons:
      - Show simple distribution statistics before and after
      - Display basic visualization if feature is numerical
      - Show value counts if feature is categorical

    - Imputation Summary Table
      - Show for each imputed feature:
        - Feature name
        - Number of values imputed
        - Method used
        - Value substituted (mean/median/mode/constant)

  - We will present this summary table before returning the modified dataframe

### Method History
 - Alpha build by Don Smith
