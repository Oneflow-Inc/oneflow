import oneflow.core.operator.op_node_signature_pb2 as op_node_signature_pb
import oneflow.framework.c_api_util as c_api_util
import oneflow._oneflow_internal.oneflow.core.operator.op_node_signature as op_node_signature_cfg
import oneflow
from google.protobuf import text_format

def Infer(op_conf, ibn2blob_object, scope_symbol_id=None):
    if scope_symbol_id is None:
        scope_symbol_id = oneflow.current_scope().symbol_id
    op_conf.scope_symbol_id = scope_symbol_id
    upstream_signature = MakeUpstreamSignature(ibn2blob_object)
    return c_api_util.InferOpConf(op_conf, upstream_signature)

def MakeUpstreamSignature(ibn2blob_object):
    upstream_signature_cfg = op_node_signature_cfg.OpNodeSignature()
    for (ibn, blob_object) in ibn2blob_object.items():
        blob_object.op_arg_blob_attr.DumpToOpNodeSignature(ibn, upstream_signature_cfg)
        blob_object.op_arg_parallel_attr.DumpToOpNodeSignature(ibn, upstream_signature_cfg)
    return text_format.Parse(str(upstream_signature_cfg), op_node_signature_pb.OpNodeSignature())