# TADPREP
*A personal project to generate a time-saving tabular data preprocessing library for Python, along with a resilient 
end-to-end CLI-based data preparation script for less-technical users.*

### Background
The TADPREP (**T**abular **A**utomated **D**ata **PREP**rocessing) library is a unified, importable package 
intended to streamline the preprocessing of tabular data. The data preparation process that is performed on tabular 
datafiles before advanced data analysis is applied is reasonably consistent and often leads to the reuse and/or 
reproduction of older, potentially-deprecated data preprocessing code. The TADPREP library helps to obviate that 
code reproduction and to save a busy data scientist time and effort.

While every analytical use case inevitably has its own particular corner cases and unique challenges, the core steps 
involved in tabular data preparation (especially assuming a naive-case approach) are consistent in form and order.
Writing fresh code to handle data preparation for each new project is labor-intensive and time-inefficient. 

As such, the TADPREP library (in addition to providing a full end-to-end script which can walk a user through data 
preprocessing) contains a number of core data-mutation methods which, when combined as per the user's specifications, 
can greatly accelerate the data preprocessing workflow.

TADPREP leverages validated data preparation libraries, extensive error handling, and a streamlined, lightweight UX 
intended to make the tabular data preparation process as smooth and low-effort as possible.

TADPREP can be imported into and run directly from the CLI or within any standard IDE.

### Using the End-to-End CLI Script
Description of what the end-to-end pipline is (with file browser I/O), its restrictions, how to use it, who it's for, 
etc. Explain how it differs from the library and when it should and should not be used. Explain the logger.

### Using the Library Methods
Explain how the methods operate on dataframes as separate entities. Explain the prep_df method which chains all the 
core TADPREP methods together and basically functions like the CLI script without the I/O process.

### Method List
List methods and what they're for in a top-line level of detail.

### Provisos
Discuss the library's limitation re: imputation, encoding, and scaling techniques. Explain how the methods should be
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