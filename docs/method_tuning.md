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

### Ideas for Development:
- Si vir manus habet operetur
