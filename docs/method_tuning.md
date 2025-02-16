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
  - HIGH PRIO: seems complete

- If `verbose=True`:
  - Print list of near-constant features (i.e. those with >= 95% matched values)
    - Implement handling of edge-case where dupes put feature at threshold of "near-constant"
  - Print list of features containing any infinite values
    - Implement count of infs per-feature
    - Could implement "location" of infs, prompted by user
  - Print list of object-type features containing *empty* strings (i.e. distinct from NULL/NaN values)
    - Could implement "location" of emptys, prompted by user

  - Define 'true_num_rows' for use with 'row_dup_rate' to avoid underestimation of duplicates
  - If implementing location of infs and emptys, ensure concise visual representation to keep interface clean
