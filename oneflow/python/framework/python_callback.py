from __future__ import absolute_import

import traceback

import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.eager.interpreter_callback as interpreter_callback
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.ofblob as ofblob


class _PythonCallback(oneflow_internal.ForeignCallback):
    def __init__(self):
        oneflow_internal.ForeignCallback.__init__(self)

    def Call(self, unique_id, of_blob_ptr):
        try:
            _WatcherHandler(unique_id, of_blob_ptr)
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def EagerInterpretCompletedOp(self, op_attribute_str, parallel_conf_str):
        try:
            interpreter_callback.InterpretCompletedOp(
                op_attribute_str, parallel_conf_str
            )
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def EagerMirroredCast(self, op_attribute_str, parallel_conf_str):
        try:
            interpreter_callback.MirroredCast(op_attribute_str, parallel_conf_str)
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def EagerCastFromMirrored(self, op_attribute_str, parallel_conf_str):
        try:
            interpreter_callback.CastFromMirrored(op_attribute_str, parallel_conf_str)
        except Exception as e:
            print(traceback.format_exc())
            raise e


def _WatcherHandler(unique_id, of_blob_ptr):
    global unique_id2handler
    assert unique_id in unique_id2handler
    handler = unique_id2handler[unique_id]
    assert callable(handler)
    handler(ofblob.OfBlob(of_blob_ptr))


unique_id2handler = {}
# static lifetime
_global_python_callback = _PythonCallback()
c_api_util.RegisterForeignCallbackOnlyOnce(_global_python_callback)
