"""Utility classes & functions for data loading. Code in this folder is mostly
used by ../dataloder.py.

A lot of multiprocessing is used in data loading, which only supports running
functions defined in global environment (py2 can't serialize static methods).
Therefore, for code tidiness we put these functions into different files in this
folder.
"""
import sys
import atexit

IS_WINDOWS = sys.platform == "win32"
MP_STATUS_CHECK_INTERVAL = 5.0
"Interval (in seconds) to check status of processes to avoid hanging in\n    multiprocessing data loading. This is mainly used in getting data from\n    another process, in which case we need to periodically check whether the\n    sender is alive to prevent hanging."
python_exit_status = False
"Whether Python is shutting down. This flag is guaranteed to be set before\nthe Python core library resources are freed, but Python may already be exiting\nfor some time when this is set.\n\nHook to set this flag is `_set_python_exit_flag`, and is inspired by a similar\nhook in Python 3.7 multiprocessing library:\nhttps://github.com/python/cpython/blob/d4d60134b29290049e28df54f23493de4f1824b6/Lib/multiprocessing/util.py#L277-L327\n"


def _set_python_exit_flag():
    global python_exit_status
    python_exit_status = True


atexit.register(_set_python_exit_flag)
from . import collate, fetch
