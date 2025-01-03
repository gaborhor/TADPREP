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


def load_file():
    pass
