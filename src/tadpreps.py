"""
This is the TADPREPS codebase. All core functions will be defined internally (so no additional files are necessary at
runtime) and the logging file will be created in the same working directory as the script itself.
"""

# Library imports
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import logging
import sys

# Set up error logging
# We persist with "%-type" formatting to preserve backward compatibility
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('tadpreps_runtime_log.log'), logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
logger.info('Initiating TADPREPS...')


def load_file() -> pd.DataFrame:
    """
    This function gets the absolute filepath from the user, performs some verification checks on the file, and if the
    checks are passed, loads the tabular data into a Pandas dataframe.
    Args:
        None. This is a nullary function.
    Returns:
        df_full (pd.DataFrame): A Pandas dataframe containing the full, unaltered dataset.
    """
    print('NOTE: TADPREPS supports only .csv and Excel files.')

    try:
        # Fetch filepath and convert to Path object
        filepath = Path(input('Enter the absolute path to your datafile: ')).resolve()

        # Error handling: Use pathlib library to check whether the supplied path is valid
        if not filepath.exists():
            logger.error(f'The supplied path "{filepath}" is invalid.')
            logger.error('Please resolve this issue and re-run TADPREPS.')
            sys.exit(1)

        # Error handling: Use pathlib library to check whether the path leads to a directory rather than a file
        if not filepath.is_file():
            logger.error(f'The supplied path "{filepath}" does not point to a single file.')
            logger.error('Please resolve this issue and re-run TADPREPS.')
            sys.exit(1)

        # Error handling: Raise error if the file exists but is of an unsupported type
        if filepath.suffix.lower() not in ['.csv', '.xls', '.xlsx']:
            logger.error('TADPREPS only supports .csv, .xls, and .xlsx files.')
            logger.error(f'The file at "{filepath}" does not appear to be of a compatible type.')
            logger.error('Please resolve this issue and re-run TADPREPS.')
            sys.exit(1)

        # Error handling: Use pathlib library to check whether the file is larger than 1GB
        size_mb = filepath.stat().st_size / (1024 ** 2)  # Fetch file size, divide by 1024^2 to get size in megabytes
        if size_mb > 1000:
            logger.error(f'File size ({size_mb} megabytes) exceeds 1 GB limit.')
            logger.error('For files of this size, consider an out-of-memory or distributed solution.')
            sys.exit(1)

        # Use pathlib library to check whether the file is .csv or Excel and load into Pandas dataframe
        if filepath.suffix.lower() == '.csv':
            file_type = 'CSV'
            df_full = pd.read_csv(filepath)
        else:
            file_type = 'Excel'
            df_full = pd.read_excel(filepath)

        logger.info(f'Successfully loaded {file_type} file at {filepath}.')
        logger.info(f'Base shape of file: {df_full.shape}')
        return df_full

    except Exception as exc:
        logger.error(f'An unexpected error occurred: {exc}')
        sys.exit(1)


def print_file_info(df_full: pd.DataFrame) -> None:
    """
    This function prints and logs use information about the full, unaltered datafile for the user.
    Args:
        df_full (pd.DataFrame): A Pandas dataframe containing the full, unaltered dataset.
    Returns:
        None. This is a void function.
    """
    # Print number of rows (instances) in file

    # Print number of columns (features) in file

    # Print names and datatypes of columns/features in file

    # Print # of instances/rows with missing values at the feature level - total count and '% rows with missing values'

    pass
