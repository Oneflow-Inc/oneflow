"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import traceback

import oneflow.python.framework.ofblob as ofblob
import oneflow_api


def GetIdForRegisteredCallback(cb):
    assert callable(cb)
    global unique_id2handler
    unique_id2handler[id(cb)] = cb
    return id(cb)


def DeleteRegisteredCallback(cb):
    global unique_id2handler
    assert id(cb) in unique_id2handler
    del unique_id2handler[id(cb)]


class PythonCallback(oneflow_api.ForeignCallback):
    def __init__(self):
        oneflow_api.ForeignCallback.__init__(self)

    def OfBlobCall(self, unique_id, of_blob_ptr):
        try:
            _WatcherHandler(unique_id, of_blob_ptr)
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def RemoveForeignCallback(self, unique_id):
        global unique_id2handler
        try:
            del unique_id2handler[unique_id]
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

    def MakeScopeSymbol(self, job_conf_str, parallel_conf_str, is_mirrored):
        try:
            return interpreter_callback.MakeScopeSymbol(
                job_conf_str, parallel_conf_str, is_mirrored
            )
        except Exception as e:
            print(traceback.format_exc())
            raise e

    def MakeParallelDescSymbol(self, parallel_conf_str):
        try:
            return interpreter_callback.MakeParallelDescSymbol(parallel_conf_str)
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
