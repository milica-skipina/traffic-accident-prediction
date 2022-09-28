"""Prepares Jupyter notebooks to increase code reusability.

Performs following tasks:
- Adds the project directory to the Python's path. This makes it easy to
import packages and modules stored in the root of the project directory.
"""

import sys
from os import path

project_dir = path.abspath(path.join('..'))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)