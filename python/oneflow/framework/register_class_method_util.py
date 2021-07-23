import oneflow.eager.eager_blob_util as eager_blob_util
import oneflow.framework.op_expr_util as op_expr_util
import oneflow.framework.functional as functional
import oneflow.framework.generator as generator
import oneflow.framework.remote_blob as remote_blob_util
import oneflow.framework.blob_trait as blob_trait
import oneflow._oneflow_internal


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
