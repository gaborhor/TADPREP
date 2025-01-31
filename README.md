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

**Note:**
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
List methods and what they're for in a top-line level of detail.

### Provisos
Discuss the library's limitations re: imputation, encoding, and scaling techniques. Explain how the methods should be
used 'as practical' - i.e. if a user wants to do fancy imputation, they can use TADPREP to trim, rename, and pull info
from the dataset, and then write their own imputation code.

### Non-Native Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

### Future Development
Any thoughts about future development