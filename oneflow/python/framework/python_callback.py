from __future__ import absolute_import

import traceback

import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.eager.interpreter_callback as interpreter_callback


class _PythonCallback(oneflow_internal.ForeignCallback):
    def __init__(self):
        oneflow_internal.ForeignCallback.__init__(self)

    def EagerInterpret(self, op_attribute_str, parallel_conf_str):
        try:
            interpreter_callback.Interpret(op_attribute_str, parallel_conf_str)
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def EagerBackwardInterpret(self, op_attribute_str, parallel_conf_str):
        try:
            interpreter_callback.BackwardInterpret(op_attribute_str, parallel_conf_str)
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def EagerCastToMirrored(self, op_attribute_str, parallel_conf_str):
        try:
            interpreter_callback.CastToMirrored(op_attribute_str, parallel_conf_str)
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def EagerCastFromMirrored(self, op_attribute_str, parallel_conf_str):
        try:
            interpreter_callback.CastFromMirrored(op_attribute_str, parallel_conf_str)
        except Exception as e:
            print(traceback.format_exc())
            raise e


# static lifetime
_global_python_callback = _PythonCallback()
c_api_util.RegisterForeignCallbackOnlyOnce(_global_python_callback)
