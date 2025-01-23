# Preparing TADPREPS to be Modularized and Pushed to PyPi

### Current Use Case for TADPREPS
Essentially, TADPREPS exists as a single, CLI-callable script which handles the import, transformation, and export of 
tabular data. It functions (as it was initially conceived to function) as an end-to-end tool which ingests a flat file
and exports its clean/mutated form as *another* flat file, which is useful and usable in all the ordinary ways we might 
expect.

However, this configuration of TADPREPS *is* limited, insofar as it isn't easily integrated into an IDE-based workflow.
The script can of course be open and used within an IDE, but it operates in generally the same manner as it does when
called from the CLI. While the initial intention of TADPREPS was to obviate the need for in-IDE data processing code,
the next logical expansion of the tool's capabilities (since its core data-processing functions are broadly decomposed)
is to further modularize the code, create methods such that certain elements/segments of the TADPREPS process can be 
called on already-loaded objects, and push the resulting system to PyPi such that it can be imported either as a full
tool or as a series of sub-tools.

In essence, the goal would be to have (among others) a method which is appropriately parameterized and callable on an
already-loaded dataframe which would handle all off the data mutation - without requiring the user to import or export
the data from a tabular source. In other words, ideally you'd have the capacity to do something similar to the 
following:
```python
# User imports a core TADPREPS method
import pandas as pd
from tadpreps import transform_df  # Or whatever it will be called

# User imports data in whatever way they wish
df_raw = pd.read_json('some_file_name')  # TADPREPS' current form doesn't support JSON imports

# User applies all parts of TADPREPS other than import/export
df_clean = tadpreps.transform_df(df_raw)  # This will be appropriately parametrized

# User performs whatever other work/analysis they require
```

### Possible Methods for the TADPREPS Package on PyPi
The basic principle is that each core data mutation function will be its own method, and there will also be a unified
method which runs the whole TADPREPS process, but without I/O, rollback, or logging. 

So, we'd have these methods as part of the TADPREPS package:
- "Unified" full-process method:
  - tadpreps.prep_df() &rarr; runs the full TADPREPS pipeline but without I/O or logging, keeps the rollback intact
- "Smaller" single-step methods:
  - tadpreps.info() &rarr; runs the print_file_info() core function
  - tadpreps.reshape() &rarr; runs the trim_file() core function
  - tadpreps.rename_and_tag() &rarr; runs the rename_features() core function
  - tadpreps.feature_stats() &rarr; runs the print_feature_stats() core function
  - tadpreps.impute() &rarr; runs the impute_missing_data() core function
  - tadpreps.encode_and_scale() &rarr; runs the encode_and_scale() core function

### Preparing and Modifying TADPREPS for PyPi Compatability
- Lorem ipsum