from __future__ import absolute_import

import traceback

import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.ofblob as ofblob


def GetIdForRegisteredCallback(cb):
    assert callable(cb)
    global unique_id2handler
    unique_id2handler[id(cb)] = cb
    return id(cb)


def DeleteRegisteredCallback(cb):
    global unique_id2handler
    assert id(cb) in unique_id2handler
    del unique_id2handler[id(cb)]


class PythonCallback(oneflow_internal.ForeignCallback):
    def __init__(self):
        oneflow_internal.ForeignCallback.__init__(self)

    def OfBlobCall(self, unique_id, of_blob_ptr):
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
# registered in the file python/framework/register_python_callback
global_python_callback = PythonCallback()

# initialized in the file python/framework/register_python_callback for avoiding import loop
interpreter_callback = None
