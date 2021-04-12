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

from oneflow.python.framework.dtype import convert_proto_dtype_to_oneflow_dtype
import oneflow.python.framework.blob_trait as blob_trait
import oneflow.python.framework.python_callback as python_callback
import oneflow.python.lib.core.async_util as async_util
import oneflow_api


@property
def dtype(self):
    return convert_proto_dtype_to_oneflow_dtype(self.get_dtype())


def numpy(self):
    return _GetPhysicalBlobBodyCache(self.blob_object)


def numpy_list(self):
    return _GetPhysicalBlobBodyCache(self.blob_object)


def RegisterMethod4EagerPhysicalBlob():
    oneflow_api.EagerPhysicalBlob.dtype = dtype
    oneflow_api.EagerPhysicalBlob.numpy = numpy
    oneflow_api.EagerPhysicalBlob.numpy_list = numpy_list


def FetchTensorBlobAsNumpyList(parallel_size, blob_object):
    def AsyncFetchBlobBody(Yield):
        fetcher = _MakeFetcherEagerBlobBodyAsNumpyFromOfBlob(Yield)

        def BuildFetchBlobBodyInstruction(builder):
            builder.FetchBlobBody(
                blob_object, python_callback.GetIdForRegisteredCallback(fetcher)
            )
            builder.InsertRemoveForeignCallbackInstruction(
                blob_object.object_id,
                python_callback.GetIdForRegisteredCallback(fetcher),
            )

        oneflow_api.deprecated.PhysicalRun(BuildFetchBlobBodyInstruction)

    return async_util.Await(parallel_size, AsyncFetchBlobBody)


def _GetPhysicalBlobHeaderCache(blob_object):
    return _FetchBlobHeader(blob_object)


def _GetPhysicalBlobBodyCache(blob_object):
    return _FetchPhysicalBlobBody(blob_object)


def _FetchBlobHeader(blob_object):
    def AsyncFetchBlobHeader(Yield):
        fetcher = _MakeFetcherEagerPhysicalBlobHeaderFromOfBlob(Yield)

        def BuildFetchBlobHeaderInstruction(builder):
            builder.FetchBlobHeader(
                blob_object, python_callback.GetIdForRegisteredCallback(fetcher)
            )
            builder.InsertRemoveForeignCallbackInstruction(
                blob_object.object_id,
                python_callback.GetIdForRegisteredCallback(fetcher),
            )

        oneflow_api.deprecated.PhysicalRun(BuildFetchBlobHeaderInstruction)

    return async_util.Await(1, AsyncFetchBlobHeader)[0]


def _FetchPhysicalBlobBody(blob_object):
    return FetchTensorBlobAsNumpyList(1, blob_object)[0]


def _MakeFetcherEagerPhysicalBlobHeaderFromOfBlob(Yield):
    def Callback(ofblob):
        Yield(
            oneflow_api.EagerPhysicalBlobHeader(
                ofblob.static_shape,
                ofblob.shape,
                oneflow_api.deprecated.GetProtoDtype4OfDtype(ofblob.dtype),
            )
        )

    return Callback


def _MakeFetcherEagerBlobBodyAsNumpyFromOfBlob(Yield):
    def FetchFromOfBlob(ofblob):
        Yield(ofblob.CopyToNdarray())

    return FetchFromOfBlob
