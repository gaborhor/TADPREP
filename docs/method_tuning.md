# Public-Facing Method Development Planning

## df_info
### Core Purpose:
Prints summary, top-level information about a dataframe to the console.

### Parameters:
- Input Pandas dataframe
- `verbose` (Boolean, default = True) which controls level of detail in output

### Returns:
- None - void method. Prints info to console.

### Current State:
- If `verbose=False`:
  - Prints number of instances (rows) in df
  - Prints number of features (columns) in df
  - Prints total number of instances with *any* missing values
  - Prints % of instances with *any* missing values as a proportion of total instances


- If `verbose=True`, the method **also** prints:
  - Names and datatypes of all features, along with memory use, etc. *(Calls Pandas' .info() method)*
  - A line of dashes on either side of the .info() output for visual separation

### Observed Bugs/Problems:
- None

### Ideas for Development:
- Print count of duplicate instances in both `verbose` states. *(This is a potential-import-error user assist.)*
- If `verbose=True`:
  - Print list of near-constant features (i.e. those with >= 90% matched values)
  - Print list of features containing any infinite values
  - Print list of object-type features containing *empty* strings (i.e. distinct from NULL/NaN values)
