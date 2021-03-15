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
from typing import Optional
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.module as module_util
import oneflow_api


@oneflow_export("random.bernoulli")
def Bernoulli(
    x: oneflow_api.BlobDesc,
    seed: Optional[int] = None,
    dtype: Optional[flow.dtype] = None,
    name: str = "Bernoulli",
) -> oneflow_api.BlobDesc:
    """This operator returns a Blob with binaray random numbers (0 / 1) from a Bernoulli distribution. 

    Args:
        x (oneflow_api.BlobDesc): The input Blob. 
        seed (Optional[int], optional): The random seed. Defaults to None.
        dtype (Optional[flow.dtype], optional): The data type. Defaults to None.
        name (str, optional): The name for the operation. Defaults to "Bernoulli".

    Returns:
        oneflow_api.BlobDesc: The result Blob. 

    For example: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def bernoulli_Job(x: tp.Numpy.Placeholder(shape=(3, 3), dtype=flow.float32),
        ) -> tp.Numpy:
            out = flow.random.bernoulli(x)
            return out


        x = np.array([[0.25, 0.45, 0.3], 
                    [0.55, 0.32, 0.13], 
                    [0.75, 0.15, 0.1]]).astype(np.float32)
        out = bernoulli_Job(x)

        # Because our random seed is not fixed, so the return value is different each time. 
        # out [[1. 0. 0.]
        #      [0. 0. 1.]
        #      [0. 0. 0.]]

    """
    assert isinstance(name, str)
    if dtype is None:
        dtype = x.dtype
    if seed is not None:
        assert name is not None
    module = flow.find_or_create_module(
        name, lambda: BernoulliModule(dtype=dtype, random_seed=seed, name=name),
    )
    return module(x)


class BernoulliModule(module_util.Module):
    def __init__(
        self, dtype: flow.dtype, random_seed: Optional[int], name: str,
    ):
        module_util.Module.__init__(self, name)
        seed, has_seed = flow.random.gen_seed(random_seed)
        self.op_module_builder = (
            flow.user_op_module_builder("bernoulli")
            .InputSize("in", 1)
            .Output("out")
            .Attr("dtype", dtype)
            .Attr("has_seed", has_seed)
            .Attr("seed", seed)
            .CheckAndComplete()
        )
        self.op_module_builder.user_op_module.InitOpKernel()

    def forward(self, x: oneflow_api.BlobDesc):
        if self.call_seq_no == 0:
            name = self.module_name
        else:
            name = id_util.UniqueStr("Bernoulli_")

        return (
            self.op_module_builder.OpName(name)
            .Input("in", [x])
            .Build()
            .InferAndTryRun()
            .SoleOutputBlob()
        )
