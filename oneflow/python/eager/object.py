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

import oneflow.python.framework.python_interpreter_util as python_interpreter_util
import oneflow.python.eager.symbol as symbol_util
import oneflow


class Object(object):
    def __init__(self, object_id, parallel_desc_symbol):
        self.object_id_ = object_id
        self.parallel_desc_symbol_ = parallel_desc_symbol

    @property
    def object_id(self):
        return self.object_id_

    @property
    def parallel_desc_symbol(self):
        return self.parallel_desc_symbol_


class BlobObject(Object):
    def __init__(self, object_id, op_arg_parallel_attr, op_arg_blob_attr, release):
        Object.__init__(self, object_id, op_arg_parallel_attr.parallel_desc_symbol)
        self.op_arg_parallel_attr_ = op_arg_parallel_attr
        self.op_arg_blob_attr_ = op_arg_blob_attr
        self.release_ = []
        if release is not None:
            self.release_.append(release)

    @property
    def op_arg_parallel_attr(self):
        return self.op_arg_parallel_attr_

    @property
    def op_arg_blob_attr(self):
        return self.op_arg_blob_attr_

    def add_releaser(self, release):
        self.release_.append(release)

    # Bind `python_interpreter_util.IsShuttingDown` early.
    # See the comments of `python_interpreter_util.IsShuttingDown`
    def __del__(self, is_shutting_down=python_interpreter_util.IsShuttingDown):
        if is_shutting_down():
            return
        for release in self.release_:
            release(self)
        self.release_ = []
