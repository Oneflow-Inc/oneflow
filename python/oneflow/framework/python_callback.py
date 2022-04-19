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
import traceback

import oneflow._oneflow_internal
import oneflow._oneflow_internal.oneflow.core.job.job_conf as job_conf_cfg
import oneflow._oneflow_internal.oneflow.core.job.placement as placement_cfg
import oneflow._oneflow_internal.oneflow.core.job.scope as scope_cfg
import oneflow._oneflow_internal.oneflow.core.operator.op_attribute as op_attribute_cfg
import oneflow.framework.ofblob as ofblob


def GetIdForRegisteredCallback(cb):
    assert callable(cb)
    global unique_id2handler
    unique_id2handler[id(cb)] = cb
    return id(cb)


def DeleteRegisteredCallback(cb):
    global unique_id2handler
    assert id(cb) in unique_id2handler
    del unique_id2handler[id(cb)]


class PythonCallback(oneflow._oneflow_internal.ForeignCallback):
    def __init__(self):
        oneflow._oneflow_internal.ForeignCallback.__init__(self)

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

    def MakeScopeSymbol(self, job_conf, parallel_conf, is_mirrored):
        try:
            return interpreter_callback.MakeScopeSymbol(
                job_conf, parallel_conf, is_mirrored
            )
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
global_python_callback = PythonCallback()
interpreter_callback = None
