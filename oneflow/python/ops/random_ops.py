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
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.module as module_util


@oneflow_export("random.bernoulli")
def Bernoulli(
    x: remote_blob_util.BlobDef,
    seed: Optional[int] = None,
    dtype: Optional[dtype_util.dtype] = None,
    name: str = "Bernoulli",
) -> remote_blob_util.BlobDef:
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
        self, dtype: dtype_util.dtype, random_seed: Optional[int], name: str,
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

    def forward(self, x: remote_blob_util.BlobDef):
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
