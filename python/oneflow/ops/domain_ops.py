import typing
import oneflow as flow
import oneflow._oneflow_internal
import oneflow.framework.id_util as id_util

def api_fused_self_attention_query_mul_key_and_value(x: oneflow._oneflow_internal.BlobDesc, head_size: int, alpha: float=1.0, name: typing.Optional[str]=None) -> oneflow._oneflow_internal.BlobDesc:
    if name is None:
        name = id_util.UniqueStr('FusedSelfAttentionQueryMulKeyAndValue_')
    op = flow.user_op_builder(name).Op('fused_self_attention_query_mul_key_and_value').Input('hidden_states', [x]).Attr('head_size', int(head_size)).Attr('alpha', float(alpha)).Output('query_mul_key').Output('value').Build()
    (qmk, v) = op.InferAndTryRun().RemoteBlobList()
    return (qmk, v)