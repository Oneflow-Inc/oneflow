import os
import sys

test_util_parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
oneflow_test_utils_dir_from_env = os.getenv("ONEFLOW_TEST_UTILS_DIR")
if oneflow_test_utils_dir_from_env:
    from pathlib import Path

    oneflow_test_utils_dir_from_env = Path(oneflow_test_utils_dir_from_env)
    test_util_parent_dir = str(oneflow_test_utils_dir_from_env.parent.absolute())
sys.path.append(test_util_parent_dir)
from test_utils.automated_test_util import *
