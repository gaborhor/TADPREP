# TADPREPS
**A personal project to generate a resilient end-to-end ML data preparation Python script for tabular data.**

### Background
The TADPREPS program (**T**abular **A**utomated **D**ata **PREP**aration **S**ystem) is a unified, command line-executable data preparation script 
intended to streamline the data preparation process that is performed on nearly every tabular datafile before any form 
of advanced data analysis is applied.

While every analytical use case inevitably has its own particular corner cases and unique challenges, the steps 
involved in tabular data preparation (especially assuming a naive-case approach) are reasonably consistent.

Writing fresh code to handle data preparation for each new project is labor-intensive and time-inefficient. Unless it 
is necessary that the data preparation process be scripted and applied within the scope of a given project, it is 
more efficient to take any flat, static datafile supplied by a customer/client/internal datasource, transform it using 
a single broadly-applicable script, and import the cleaned, processed data into the project.

TADPREPS leverages standard data preparation libraries, extensive error handling and logging, and a streamlined, 
lightweight UX intended to make the tabular data preparation process as smooth and low-effort as possible.

It can be run directly from the command line or within any standard IDE. The log file created by the script is 
stored in the same directory as the exported datafile for co-location purposes.

### Use Case/General Functionality
The TADPREPS tool is designed to operate as a sort of "Swiss Army Knife" program which can handle the most common
tabular datafiles and perform the most common data transformations which are needed in the majority of data science
tasks. The tool is user-interactive: it first fetches and loads the file, allows the user to delete or change data,
subsample instances to reduce file size, provides relevant visualizations for each feature, and then takes the user
through a step-by-step process of applying encoding and scaling to the features in the data. It ends by providing
a few different options for exporting the cleaned, transformed data.

*NOTE: A more in-depth explanation of the exact procedural flow of the script can be found in the
'data_prep_workflow.md' document in the 'docs' folder of this repository.*

In terms of the daily work of a data scientist, TADPREPS is designed to integrate into the workflow as follows:
1. A data scientist is provided with a flat file of relevant data, either externally by a client or by running a database query of some sort.
2. The data scientist moves the data into the appropriate location.
3. The data scientist runs TADPREPS on the file, and sequentially performs file trimming, feature modification, EDA, and data transformation.
4. The cleaned file is exported to a location of the data scientist's choosing in one of several export formats supported by TADPREPS. 
5. The cleaned file can be directly imported into a new codebase for further analysis, leaving the advanced work uncluttered by initial data preparation.

### Knowledge Assumed
TADPREPS is **not** a general-populace tool. It is not intended for use by people without data science training, because 
it asks the user to make decisions regarding data transformations based on descriptive statistics and plots which are 
not explained by the program. It assumes (for example) that the user knows how to run a Python program, how to locate 
and pass an absolute filepath, what ordinal data are, how to read a histogram, what sub-sampling is, etc.

This script is **not** intended to be a teaching tool about how tabular data preparation works. It is a process-optimized
fast-decision tool which assumes that the user knows what they're doing, what they're looking at, and which choices are
correct given the information supplied by the program. It is a tool designed to be a part of a trained data scientist's
ordinary workflow.

### Technical Resources Assumed
The TADPREPS tool makes some basic assumptions about the technical resources available to the user, all of which
seem (at least to me) to be reasonable in the context of the tools and packages that an intended user 
(i.e. a Data Scientist) is likely to already have available to them. 

The tool assumes that:
1. The user has Python installed.
2. The user knows how to use a CLI.
3. The user has write-level privileges on their machine.
4. The user has the most common data science libraries installed.
5. The user's machine is sufficiently powerful to run various Scikit-Learn and Seaborn methods without crashing.

### Future Development
While future development needs are unclear at this point, there are a couple of things I have my eye on, including:
1. Building roll-back functionality into the pipeline so the user can return to earlier stages in the data transformation process.
2. Building in a Dask-centric capability to work with very large tabular files outside of active memory.
3. Building in a functionality which allows the user to automatically test the features for the statistical assumptions of linear modeling
