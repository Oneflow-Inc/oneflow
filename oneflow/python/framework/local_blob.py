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


class LocalMirroredTensor(object):
    def __init__(self, ndarray_list, is_dynamic, concat_axis=None):
        self.ndarray_list_ = ndarray_list
        self.is_dynamic_ = is_dynamic
        self.concat_axis_ = concat_axis
        self.ndarray_ = None
        if not is_dynamic:
            if len(self.ndarray_list_) == 1:
                self.ndarray_ = self.ndarray_list_[0]
            elif concat_axis is not None:
                self.ndarray_ = np.concatenate(self.ndarray_list_, axis=concat_axis)
            else:
                # do nothing
                pass

    @property
    def is_dynamic(self):
        return self.is_dynamic_

    def ndarray_list(self):
        print(
            "WARNING:",
            "LocalMirroredTensor.ndarray_list is deprecated, please use LocalMirroredTensor.numpy_list\n",
            traceback.format_stack()[-2],
        )
        return self.numpy_list()

    def numpy_list(self):
        return self.ndarray_list_

    def ndarray(self):
        print(
            "WARNING:",
            "LocalMirroredTensor.ndarray is deprecated, please use LocalMirroredTensor.numpy\n",
            traceback.format_stack()[-2],
        )
        return self.numpy()

    def numpy(self, parallel_id=None):
        if parallel_id is None:
            assert self.ndarray_ is not None
            return self.ndarray_
        else:
            assert parallel_id >= 0
            assert len(self.ndarray_list_) > parallel_id
            return self.ndarray_list_[parallel_id]

    def parallel_num(self):
        return len(self.ndarray_list_)

    def __getattr__(self, attr):
        return getattr(self.numpy(), attr)


class LocalMirroredTensorList(object):
    def __init__(self, ndarray_lists=None):
        assert isinstance(ndarray_lists, (list, tuple))
        for ndarray_list in ndarray_lists:
            assert isinstance(ndarray_list, (list, tuple))
            assert all(isinstance(ndarray, np.ndarray) for ndarray in ndarray_list)
        self.ndarray_lists_ = ndarray_lists

    def ndarray_lists(self):
        print(
            "WARNING:",
            "LocalMirroredTensorList.ndarray_lists is deprecated, please use LocalMirroredTensorList.numpy_lists",
        )
        return self.numpy_lists()

    def numpy_lists(self):
        return self.ndarray_lists_

    def numpy_list(self, parallel_id=None):
        if parallel_id is None:
            assert len(self.ndarray_lists_) == 0
            return self.ndarray_lists_[0]
        else:
            assert parallel_id >= 0
            assert len(self.ndarray_lists_) > parallel_id
            return self.ndarray_lists_[parallel_id]

    def parallel_num():
        return len(self.ndarray_lists_)


def MakeLocalBlob(ndarray_lists, consistent_blob):
    assert isinstance(consistent_blob, oneflow_api.ConsistentBlob), type(
        consistent_blob
    )
    if consistent_blob.is_tensor_list:
        return LocalMirroredTensorList(ndarray_lists)
    assert len(ndarray_lists) == 1
    return LocalMirroredTensor(
        ndarray_lists[0],
        is_dynamic=consistent_blob.is_dynamic,
        concat_axis=consistent_blob.split_axis,
    )


def MergeLocalBlobs(local_blob_list, mirrored_blob):
    assert isinstance(mirrored_blob, oneflow_api.MirroredBlob)
    if mirrored_blob.is_tensor_list:
        for local_blob in local_blob_list:
            assert type(local_blob) is LocalMirroredTensorList
        return LocalMirroredTensorList([x.numpy_lists()[0] for x in local_blob_list])
    # NOTE(chengcheng): concat_axis=split_axis just to be sure. Will delete in multi-client.
    return LocalMirroredTensor(
        [x.numpy_list()[0] for x in local_blob_list],
        is_dynamic=mirrored_blob.is_dynamic,
        concat_axis=mirrored_blob.split_axis,
    )


def MakeLocalBlob4EagerBlob(eager_blob):
    assert isinstance(eager_blob, oneflow_api.EagerBlobTrait)
    if eager_blob.is_tensor_list:
        return LocalMirroredTensorList(eager_blob.numpy_list())
    elif isinstance(eager_blob, oneflow_api.EagerMirroredBlob):
        # NOTE(chengcheng): concat_axis=split_axis just to be sure. Will delete in multi-client.
        return LocalMirroredTensor(
            [eager_blob.numpy(i) for i in range(eager_blob.numpy_size())],
            is_dynamic=eager_blob.is_dynamic,
            concat_axis=eager_blob.split_axis,
        )
    elif isinstance(eager_blob, oneflow_api.EagerConsistentBlob):
        return LocalMirroredTensor(
            [eager_blob.numpy()], is_dynamic=False, concat_axis=0
        )
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
        return [x.numpy() if isinstance(x, LocalMirroredTensor) else x for x in args]

    return lambda self, *args: getattr(self.numpy(), field_name)(
        *ConvertOtherArgs(args)
    )


for field_name in dir(np.ndarray):
    if field_name.startswith("__") == False:
        continue
    if field_name in non_override_field:
        continue
    if hasattr(LocalMirroredTensor, field_name) == False:
        setattr(LocalMirroredTensor, field_name, MakeBlobMethod(field_name))
