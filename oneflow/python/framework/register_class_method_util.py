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
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.eager_blob_util as eager_blob_util
import oneflow.python.framework.op_expr_util as op_expr_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.blob_trait as blob_trait
import oneflow_api


def RegisterMethod4Class():
    op_expr_util.RegisterMethod4UserOpExpr()

    eager_blob_util.RegisterMethod4EagerPhysicalBlob()

    blob_trait.RegisterBlobOperatorTraitMethod(oneflow_api.EagerPhysicalBlob)
    blob_trait.RegisterBlobOperatorTraitMethod(oneflow_api.ConsistentBlob)
    blob_trait.RegisterBlobOperatorTraitMethod(oneflow_api.MirroredBlob)

    remote_blob_util.RegisterMethod4EagerBlobTrait()
    remote_blob_util.RegisterMethod4LazyConsistentBlob()
    remote_blob_util.RegisterMethod4LazyMirroredBlob()
    remote_blob_util.RegisterMethod4EagerConsistentBlob()
    blob_cache_util.RegisterMethodAndAttr4BlobCache()
