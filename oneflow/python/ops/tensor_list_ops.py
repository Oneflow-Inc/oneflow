import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("tensor_list_to_tensor_buffer")
def tensor_list_to_tensor_buffer(input, name=None):
    if name is None:
        name = id_util.UniqueStr("TensorListToBuffer_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_list_to_tensor_buffer_conf, "in", input.unique_name)
    setattr(op_conf.tensor_list_to_tensor_buffer_conf, "out", "out")
    compile_context.CurJobAddOp(op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("tensor_buffer_to_tensor_list")
def tensor_buffer_to_tensor_list(input, shape, dtype, name=None):
    if name is None:
        name = id_util.UniqueStr("TensorBufferToList_")

    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "in", input.unique_name)
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "out", "out")
    op_conf.tensor_buffer_to_tensor_list_conf.shape.dim[:] = list(shape)
    setattr(op_conf.tensor_buffer_to_tensor_list_conf, "data_type", dtype)
    compile_context.CurJobAddOp(op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("tensor_list_split")
def tensor_list_split(input_tensor_list, name=None):
    if name is None:
        name = id_util.UniqueStr("TensorListSplit_")

    output_size = input_tensor_list.shape[0]
    op_conf = op_conf_util.OperatorConf()
    setattr(op_conf, "name", name)
    setattr(op_conf.tensor_list_split_conf, "in", input_tensor_list.unique_name)
    op_conf.tensor_list_split_conf.out.extend(
        ["out_{}".format(i) for i in range(output_size)]
    )
    compile_context.CurJobAddOp(op_conf)
    ret = []
    for i in range(output_size):
        out_lbi = logical_blob_id_util.LogicalBlobId()
        setattr(out_lbi, "op_name", op_conf.name)
        setattr(out_lbi, "blob_name", "out_{}".format(i))
        ret.append(remote_blob_util.RemoteBlob(out_lbi))
    return tuple(ret)
