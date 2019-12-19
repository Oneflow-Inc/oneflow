import unittest
import inspect
from oneflow.python.oneflow_export import oneflow_export
import oneflow 

@oneflow_export('unittest.TestCase')
class TestCase(unittest.TestCase):
    def tearDown(self):
        oneflow.clear_default_session()

@oneflow_export('unittest.run')
def run():
    unittest.main()

@oneflow_export('unittest.run_if_this_is_main')
def run_if_this_is_main():
    if _GetCallerModuleName() == '__main__':
        unittest.main()

def _GetCallerModuleName():
    frame = inspect.stack()[2]
    return inspect.getmodule(frame[0]).__name__
