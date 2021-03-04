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

import numpy as np
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow_api
import traceback


class LocalBlob(object):
    # TODO(chengcheng): maybe not need LocalBlob.
    def __init__(self, ndarray, is_dynamic):
        self.ndarray_ = ndarray
        self.is_dynamic_ = is_dynamic

    @property
    def is_dynamic(self):
        return self.is_dynamic_

    def ndarray_list(self):
        print(
            "WARNING:",
            "LocalBlob.ndarray_list is deprecated, please use LocalBlob.numpy()\n",
            traceback.format_stack()[-2],
        )
        return self.numpy_list()

    def numpy_list(self):
        print(
            "WARNING:",
            "LocalBlob.numpy_list() is deprecated, it will return [LocalBlob.numpy()].",
            "please use LocalBlob.numpy()\n",
            traceback.format_stack()[-2],
        )
        return [self.numpy()]

    def ndarray(self):
        print(
            "WARNING:",
            "LocalBlob.ndarray is deprecated, please use LocalBlob.numpy()\n",
            traceback.format_stack()[-2],
        )
        return self.numpy()

    def numpy(self, parallel_id=None):
        assert parallel_id is None or parallel_id == 0
        return self.ndarray_

    def parallel_num(self):
        return 1

    def __getattr__(self, attr):
        return getattr(self.numpy(), attr)


def MakeLocalBlob4EagerBlob(eager_blob):
    # TODO(chengcheng): refactor eager local blob.
    assert isinstance(eager_blob, oneflow_api.EagerBlobTrait)
    if isinstance(eager_blob, oneflow_api.EagerMirroredBlob):
        assert eager_blob.numpy_size() == 1
        return LocalBlob(eager_blob.numpy(), is_dynamic=eager_blob.is_dynamic,)
    elif isinstance(eager_blob, oneflow_api.EagerConsistentBlob):
        return LocalBlob(eager_blob.numpy(), is_dynamic=False)
    else:
        raise NotImplementedError


non_override_field = set(
    [
        "__class__",
        "__doc__",
        "__new__",
        "__init__",
        "__del__",
        "__call__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__dir__",
        "__get__",
        "__set__",
        "__delete__",
    ]
)


def MakeBlobMethod(field_name):
    def ConvertOtherArgs(args):
        return [x.numpy() if isinstance(x, LocalBlob) else x for x in args]

    return lambda self, *args: getattr(self.numpy(), field_name)(
        *ConvertOtherArgs(args)
    )


for field_name in dir(np.ndarray):
    if field_name.startswith("__") == False:
        continue
    if field_name in non_override_field:
        continue
    if hasattr(LocalBlob, field_name) == False:
        setattr(LocalBlob, field_name, MakeBlobMethod(field_name))
