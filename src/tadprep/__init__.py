"""TADPREP - Interactive Data Preparation Tool"""

# Import public-facing methods that users will access
from .package import (
    prep_df,
    info,
    reshape,
    rename_and_tag,
    feature_stats,
    impute,
    encode_and_scale
)

# Define exposure if "from tadprep import *" is called
__all__ = [
    'prep_df',
    'info',
    'reshape',
    'rename_and_tag',
    'feature_stats',
    'impute',
    'encode_and_scale',
]

# Package metadata
__version__ = '0.1.0'
__author__ = 'Donald Smith'


def main():
    """Entry point for running the full interactive version of the pipeline"""
    from .tadprep_interactive import main as interactive_main
    interactive_main()


if __name__ == '__main__':
    main()
