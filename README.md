# TADPREP
*A personal project to generate a time-saving tabular data preprocessing library for Python, along with a resilient 
end-to-end CLI-based data preparation script for less-technical users.*

### Background
The TADPREP (**T**abular **A**utomated **D**ata **PREP**rocessing) library is a unified, importable package 
intended to streamline the preprocessing of tabular data. The data preparation process that is performed on tabular 
datafiles before advanced data analysis is reasonably consistent and often leads to the reuse and/or 
reproduction of older, potentially-deprecated data preprocessing code. The TADPREP library helps to obviate that 
code reproduction and to save a busy data scientist time and effort.

While every analytical task inevitably has its own particular corner cases and unique challenges, the core steps 
involved in tabular data preparation (especially assuming a naive-case approach) are consistent in form and order.
Writing fresh code to handle data preparation for each new project is labor-intensive and time-inefficient. 

As such, the TADPREP library (in addition to providing a full end-to-end script which can walk a user through data 
preprocessing) contains a number of core data-mutation methods which, when combined as per the user's specifications, 
can greatly accelerate their data preprocessing workflow.

TADPREP leverages validated data preparation libraries, extensive error handling, and a streamlined, lightweight UX 
intended to make the tabular data preparation process as smooth and low-effort as possible.

TADPREP can be imported into any standard IDE. The end-to-end pipeline script can be run directly from the CLI.

### Using the End-to-End Pipeline Script on the CLI
The TADPREP library contains an end-to-end CLI-executable data preprocessing pipeline script. Once the TADPREPS library
is installed on your machine, this interactive script can be activated by running the following command on the CLI:
```bash
tadprep
```

Alternately, if that simple execution method fails, you may run this more explicit call on the CLI:
```bash
python -m tadprep
```

This script serves as a more fully interactive, "hand-holding" process which will walk a user through the ingestion of
a tabular file, the mutation of the data in the file, and the export of the transformed data into a new static file. 

The script accepts .csv, .tsv, and Excel-type files as input, and supports the creation of .csv, .tsv, Excel, Pickle, 
Parquet, and JSON files during the file-export process.

The end-to-end script has two primary use cases:
1. It can be used by less programming-proficient people who need some more help with the process.
2. It can be used by anyone who is engaged in a straightforward "flat to flat" data-processing task.

The script has extensive error handling, and in addition to its console output, it generates a summary runtime
log which describes the major decisions made during the process. (The log also documents any major errors that occur.)
The log file is automatically saved in the same directory as the final exported datafile.

**Notes:**
- The interactive pipeline script can only ingest locally-stored files.
- Users may identify the file to import and the export save directory either by passing an absolute filepath or by 
using a file browser window.
- In order for the script to run properly, users will need write-level permissions on their machines.
- The script includes a dependency check to ensure that all dependent libraries are installed and operable.
- The script is not readily "customizable," and is intended for use in ordinary data processing tasks. 
- Users who wish to segment or extend the core data mutation steps should import the package into an IDE and use the 
supplied methods to augment their own task-specific data-processing code.

### Using the Library Methods
The TADPREP library contains a series of callable methods, each of which represent specific subsets/segments of the full
data preparation process as conducted in the interactive CLI pipeline script. 

These methods are intended for in-IDE use such that a user may import a dataset into a Pandas dataframe by whatever 
means they deem appropriate, and then apply the data mutation methods as desired to that dataframe.

These methods are broadly parallel in form and function to various Pandas methods. Therefore, the process of using them
ought to be relatively comprehensible to anyone used to working with the Pandas library.

The methods provide a great deal of visual parsing of console output and offer explanations/warnings which are relevant 
to each step. Each method has a boolean `verbose` parameter (which is `True` by default), but can be set to `False` when 
the method is called to simplify printed output and suppress explanations.

Additionally, some methods have a further boolean `skip_warnings` parameter (which is `True` by default), but can be 
set to `False` to allow the user to skip past any mathematics- or best-practice related warnings relevant to a given 
data preprocessing step, e.g. warnings about summoning the curse of dimensionality when encoding high-cardinality 
features.

Therefore, experienced data scientists who have an excellent sense of their dataset and their intentions may prefer to
call the methods with `verbose=False` and `skip_warnings=True` to have an efficient, "quiet" user experience and to
move the data preparation process along as quickly as possible.

**EXAMPLE USAGE:**
```python
import pandas as pd
import tadprep as tp
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression

# Create database connection
engine = create_engine('postgresql://username:password@localhost:5432/mydatabase')

# Run query and store results in Pandas dataframe
df_raw = pd.read_sql_table(
    table_name='sales_data',
    con=engine,
    schema='public',
    columns=['sale_id', 
             'rep_id', 
             'product_name', 
             'quantity', 
             'sale_date', 
             'customer_satisfaction'],
    index_col='sale_id'
)

# Display general dataframe information using TADPREP
tp.file_info(df_raw, verbose=True)

# Reshape data using TADPREP
df_reshape = tp.reshape(df_raw, verbose=False)

# Perform imputation using TADPREP
df_imputed = tp.impute(df_reshape, verbose=False, skip_warnings=True)

# Run basic regression on reshaped data using 'customer_satisfaction' as the target feature
X = df_imputed.drop(['customer_satisfaction'], axis=1)
y = df_imputed['customer_satisfaction']

model = LinearRegression()
model.fit(X, y)
```

**A Note on the `prep_df` Method:**

The `prep_df` method essentially runs the same interactive data mutation pipeline as the CLI script, but it does not
include the steps pertaining to the file I/O process. If the user wants to perform all of the data mutation steps
present in the full CLI script, but wants to stay within their IDE and use data objects they have already loaded, this
method will allow them to do so. 

This method *does* offer users the option to select which of the core 
data-processing steps will be performed, but it offers the methods to the user in strict sequential order. This method
is therefore lower-effort, but less flexible.

### Full Library Method List

TADPREP provides a suite of interactive methods for data preparation, each focused on a specific aspect of the 
data preparation process:

#### Core Pipeline Method

`prep_df(df)`
- Runs the complete TADPREP pipeline on an existing DataFrame
- Guides users through all preparation steps interactively
- Returns the fully prepared DataFrame
- Ideal for users who want to prepare their data in a single cohesive workflow
- This method (in essence) runs the full interactive script without the file I/O process

#### Individual Data Preparation Methods

`df_info(df, verbose=True)`
- Displays comprehensive information about DataFrame structure and contents
- Shows feature counts, data types, and missingness statistics
- Helps users understand their data before making preparation decisions
- Set `verbose=False` for condensed output

`reshape(df, verbose=True)`
- Handles missing value deletion and feature deletion
- Allows random sub-setting of data
- Returns reshaped DataFrame
- Set `verbose=False` for minimal process output

`rename_and_tag(df, verbose=True, tag_features=False)`
- Facilitates feature renaming
- Optionally tags features as ordinal or target features when `tag_features=True`
- Returns DataFrame with updated feature names
- Set `verbose=False` for streamlined interaction

`feature_stats(df, verbose=True, summary_stats=False)`
- Analyzes features by type (categorical, numerical)
- Displays missingness information and appropriate descriptive statistics
- Set `summary_stats=True` to include aggregate statistics by feature type
- Set `verbose=False` for key statistics only

`impute(df, verbose=True, skip_warnings=False)`
- Handles missing value imputation using mean, median, or mode
- Provides guidance on appropriate imputation methods
- Returns DataFrame with imputed values
- Set `skip_warnings=True` to bypass missingness threshold warnings

`encode(df, features_to_encode=None, verbose=True, skip_warnings=False)`
- Encodes categorical features using One-Hot or Dummy encoding
- Can be passed a list of specific features to encode via the `features_to_encode` parameter
- Auto-detects categorical features if `features_to_encode` is None
- Returns DataFrame with encoded features
- Set `skip_warnings=True` to bypass cardinality and null value warnings

`scale(df, features_to_scale=None, verbose=True, skip_warnings=False)`
- Scales numerical features using Standard, Robust, or MinMax scaling
- Can be passed a list of specific features to scale via the `features_to_scale` parameter
- Auto-detects numerical features if `features_to_scale` is None
- Returns DataFrame with scaled features
- Set `skip_warnings=True` to bypass distribution and outlier warnings

#### Notes on Method Usage

- All methods include interactive prompts to guide users through the preparation process
- Methods with `verbose` parameters allow control over output detail level
- Each method can be used independently or as part of the full pipeline via `prep_df()`
- When used without re-assignment, the methods preserve the original DataFrame and return a new dataframe with 
the applied transformations. However, the user must take care that potential overwrites are handled correctly:
```python
# Prevents overwrite of original dataframe by returning new dataframe object
df_reshaped = tp.reshape(df, verbose=False)

# Overwrites original dataframe via re-assignment
df = tp.reshape(df, verbose=False)
```

### Provisos and Limitations
TADPREP is designed to be a practical, first-principle tool for data preprocessing tasks. 
While it provides interactive guidance throughout the data preparation process, it intentionally implements only 
fundamental techniques for certain operations:

#### Imputation
- Supports only simple imputation methods: mean, median, and mode
- Does not implement more sophisticated approaches like:
  - Multiple imputation
  - K-Nearest Neighbors imputation
  - Regression-based imputation
  - Machine learning models for imputation

#### Encoding
- Limited to One-Hot and Dummy encoding for categorical features
- Does not implement advanced encoding techniques such as:
  - Target encoding
  - Weight of Evidence encoding
  - Feature hashing
  - Embedding techniques

#### Scaling
- Implements three common scaling methods: Standard (Z-score), Robust, and MinMax scaling
- Does not include more specialized techniques like:
  - Quantile transformation
  - Power transformations (e.g., Box-Cox, Yeo-Johnson)
  - Custom scaling methods

#### Intended Use

TADPREP is most effectively used as part of a larger data preparation workflow.
1. Use TADPREP for initial data exploration and basic preparation, such as:
   - Understanding feature distributions and relationships
   - Identifying and handling missing values
   - Basic feature renaming and organization
   - Simple transformations using supported methods

2. For more complex operations:
   - Use TADPREP to prepare and clean your data
   - Export a partially prepared dataset
   - Apply your own custom transformations using specialized libraries like scikit-learn, feature-engine, 
   or category_encoders

Using the library in a deliberate, as-needed manner allows you to leverage TADPREP's interactive guidance for basic 
tasks while maintaining the flexibility to implement more sophisticated methods as required for your specific use case.

For example, if you need to use advanced imputation techniques:
```python
# 1. Use TADPREP for initial preparation
import pandas as pd
import tadprep as tp
df_initial = tp.prep_df(df)  # Handle basic cleaning and preparation

# 2. Import and apply custom imputation
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df_initial),
    columns=df_initial.columns
)
```

### Non-Native Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

### Future Development
Any thoughts about future development