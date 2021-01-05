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
import oneflow as flow
from typing import Tuple
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow_api


@oneflow_export("experimental.ssp_variable_proxy")
def ssp_variable_proxy(
    var: oneflow_api.BlobDesc, buffer_size: int = 1, name=None
) -> Tuple[oneflow_api.BlobDesc, oneflow_api.BlobDesc]:
    r""" return ref_blob, value_blob """
    if name is None:
        name = id_util.UniqueStr("SspVariableProxy_")
    blob_dict = (
        flow.user_op_builder(name)
        .Op("ssp_variable_proxy")
        .Input("var", [var])
        .Output("ref")
        .Output("value")
        .Attr("buffer_size", buffer_size)
        .Build()
        .InferAndTryRun()
        .RemoteBlobDict()
    )
    return blob_dict["ref"][0], blob_dict["value"][0]
