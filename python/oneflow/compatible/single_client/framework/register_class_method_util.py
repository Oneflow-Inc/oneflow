import oneflow._oneflow_internal
from oneflow.compatible.single_client.python.eager import (
    eager_blob_util as eager_blob_util,
)
from oneflow.compatible.single_client.python.framework import blob_trait as blob_trait
from oneflow.compatible.single_client.python.framework import functional as functional
from oneflow.compatible.single_client.python.framework import generator as generator
from oneflow.compatible.single_client.python.framework import (
    op_expr_util as op_expr_util,
)
from oneflow.compatible.single_client.python.framework import (
    remote_blob as remote_blob_util,
)


def RegisterMethod4Class():
    op_expr_util.RegisterMethod4UserOpExpr()
    functional.RegisterFunctionalApis()
    eager_blob_util.RegisterMethod4EagerPhysicalBlob()
    blob_trait.RegisterBlobOperatorTraitMethod(
        oneflow._oneflow_internal.EagerPhysicalBlob
    )
    blob_trait.RegisterBlobOperatorTraitMethod(oneflow._oneflow_internal.ConsistentBlob)
    blob_trait.RegisterBlobOperatorTraitMethod(oneflow._oneflow_internal.MirroredBlob)
    remote_blob_util.RegisterMethod4EagerBlobTrait()
    remote_blob_util.RegisterMethod4LazyConsistentBlob()
    remote_blob_util.RegisterMethod4LazyMirroredBlob()
    remote_blob_util.RegisterMethod4EagerConsistentBlob()
