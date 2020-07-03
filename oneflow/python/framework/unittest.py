from __future__ import absolute_import

import imp
import inspect
import os
import unittest

import oneflow
from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("unittest.register_test_cases")
def register_test_cases(
    scope, directory, filter_by_num_nodes, base_class=unittest.TestCase
):
    def FilterTestPyFile(f):
        return (
            os.path.isfile(os.path.join(directory, f))
            and f.endswith(".py")
            and f.startswith("test")
        )

    def FilterMethodName(module, name):
        method = getattr(module, name)
        return (
            name.startswith("test")
            and callable(method)
            and filter_by_num_nodes(_GetNumOfNodes(method))
        )

    onlytest_files = [f for f in os.listdir(directory) if FilterTestPyFile(f)]
    for f in onlytest_files:
        class_name = f[0:-3]
        module = imp.load_source(class_name, os.path.join(directory, f))
        test_func_names = [
            name for name in dir(module) if FilterMethodName(module, name)
        ]
        method_dict = {k: getattr(module, k) for k in test_func_names}
        scope[class_name] = type(
            class_name, (_ClearDefaultSession, base_class), method_dict
        )


@oneflow_export("unittest.num_nodes_required")
def num_nodes_required(num_nodes):
    def Decorator(f):
        f.__oneflow_test_case_num_nodes_required__ = num_nodes
        return f

    return Decorator


class _ClearDefaultSession(object):
    def setUp(self):
        oneflow.clear_default_session()


def _GetNumOfNodes(func):
    if hasattr(func, "__oneflow_test_case_num_nodes_required__") == False:
        return 1
    return getattr(func, "__oneflow_test_case_num_nodes_required__")
