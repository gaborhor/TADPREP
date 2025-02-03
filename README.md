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
Explain how the methods operate on dataframes as separate entities. Explain the prep_df method which chains all the 
core TADPREP methods together and basically functions like the CLI script without the I/O process.

### Method List

TADPREP provides a suite of interactive methods for data preparation, each focused on a specific aspect of the preparation process:

#### Core Pipeline Method

`prep_df(df)`
- Runs the complete TADPREP pipeline on an existing DataFrame
- Guides users through all preparation steps interactively
- Returns the fully prepared DataFrame
- Ideal for users who want to prepare their data in a single cohesive workflow
- This method (in essence) runs the full interactive script without the file I/O process

#### Individual Data Preparation Methods

`file_info(df, verbose=True)`
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
- Analyzes features by type (categorical, ordinal, numerical)
- Displays missingness information and appropriate descriptive statistics
- Set `summary_stats=True` to include aggregate statistics by feature type
- Set `verbose=False` for key statistics only

`impute(df, verbose=True, skip_warnings=False)`
- Handles missing value imputation using mean, median, or mode
- Provides guidance on appropriate imputation methods
- Returns DataFrame with imputed values
- Set `skip_warnings=False` to bypass missingness threshold warnings

`encode_and_scale(df, cat_cols, ord_cols, num_cols)`
- Encodes categorical features using One-Hot or Dummy encoding
- Scales numerical features using Standard, Robust, or MinMax scaling
- Returns DataFrame with transformed features
- Requires lists of categorical, ordinal, and numerical column names

#### Notes on Method Usage

- All methods include interactive prompts to guide users through the preparation process
- Methods with `verbose` parameters allow control over output detail level
- Each method can be used independently or as part of the full pipeline via `prep_df()`
- Methods preserve the original DataFrame and return a new copy with applied transformations

### Provisos and Limitations
TADPREP is designed to be a practical, educational tool for basic data preparation tasks. 
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