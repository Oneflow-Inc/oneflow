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


class LocalTensor(object):
    def __init__(self, ndarray, is_dynamic):
        self.ndarray_ = ndarray
        self.is_dynamic_ = is_dynamic

    @property
    def is_dynamic(self):
        return self.is_dynamic_

    def ndarray_list(self):
        print(
            "WARNING:",
            "LocalTensor.ndarray_list is deprecated, please use LocalTensor.numpy()\n",
            traceback.format_stack()[-2],
        )
        return self.numpy_list()

    def numpy_list(self):
        print(
            "WARNING:",
            "LocalTensor.numpy_list is deprecated, it will return [LocalTensor.numpy()].",
            "please use LocalTensor.numpy()\n",
            traceback.format_stack()[-2],
        )
        return [self.ndarray_]

    def ndarray(self):
        print(
            "WARNING:",
            "LocalTensor.ndarray is deprecated, please use LocalTensor.numpy()\n",
            traceback.format_stack()[-2],
        )
        return self.numpy()

    def numpy(self):
        return self.ndarray_

    def __getattr__(self, attr):
        return getattr(self.numpy(), attr)


def MakeLocalBlob(ndarray, consistent_blob):
    # NOTE(chengcheng): tmp support mirror blob using LocalTensor in 1 device.
    # assert isinstance(consistent_blob, oneflow_api.ConsistentBlob), type(
    #     consistent_blob
    # )
    return LocalTensor(ndarray, is_dynamic=consistent_blob.is_dynamic,)


def MergeLocalBlobs(local_blob_list, mirrored_blob):
    assert isinstance(mirrored_blob, oneflow_api.MirroredBlob)
    return LocalTensor(
        local_blob_list[0].numpy(),
        is_dynamic=mirrored_blob.is_dynamic,
        # NOTE(chengcheng): concat_axis=split_axis just to be sure. Will delete in multi-client.
        # concat_axis=mirrored_blob.split_axis,
    )


def MakeLocalBlob4EagerBlob(eager_blob):
    assert isinstance(eager_blob, oneflow_api.EagerBlobTrait)
    if isinstance(eager_blob, oneflow_api.EagerMirroredBlob):
        # NOTE(chengcheng): concat_axis=split_axis just to be sure. Will delete in multi-client.
        return LocalTensor(
            [eager_blob.numpy(i) for i in range(eager_blob.numpy_size())],
            is_dynamic=eager_blob.is_dynamic,
            # concat_axis=eager_blob.split_axis,
        )
    elif isinstance(eager_blob, oneflow_api.EagerConsistentBlob):
        return LocalTensor([eager_blob.numpy()], is_dynamic=False, concat_axis=0)
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
        return [x.numpy() if isinstance(x, LocalTensor) else x for x in args]

    return lambda self, *args: getattr(self.numpy(), field_name)(
        *ConvertOtherArgs(args)
    )


for field_name in dir(np.ndarray):
    if field_name.startswith("__") == False:
        continue
    if field_name in non_override_field:
        continue
    if hasattr(LocalTensor, field_name) == False:
        setattr(LocalTensor, field_name, MakeBlobMethod(field_name))
