@oneflow_export("math.reduce_sum")
def reduce_sum(input_tensor, axis=None, keepdims=False, name=None):
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf, "name", name if name is not None else id_util.UniqueStr("ReduceSum_")
    )
    setattr(op_conf.reduce_sum_conf, "in", input_tensor.logical_blob_name)
    setattr(op_conf.reduce_sum_conf, "out", "out")
    if axis is not None:
        assert isinstance(axis, list) or isinstance(axis, tuple)
        op_conf.reduce_sum_conf.axis[:] = list(axis)
    setattr(op_conf.reduce_sum_conf, "keep_dims", keepdims)
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
