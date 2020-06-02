from __future__ import absolute_import
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.framework.placement_context as placement_ctx

from oneflow.python.oneflow_export import oneflow_export
import oneflow

@oneflow_export("copy")
def api_copy(x, name=None):
    return enable_if.unique(lazy_copy, eager_copy)(x, name=name)

@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def lazy_copy(x, name=None):
    op_conf, lbi = _CopyOpConfAndLbi(x, name=name)
    compile_context.CurJobAddOp(op_conf)
    return remote_blob_util.RemoteBlob(lbi)

@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def eager_copy(x, name=None):
    op_conf, lbi = _CopyOpConfAndLbi(x, name=name)
    compile_context.CurJobAddMirroredOp(op_conf)
    vm_util.LogicalRun(_MakeCopyInstructionBuilderFunction(x, op_conf))
    return remote_blob_util.EagerLogicalBlob(lbi)

def _CopyOpConfAndLbi(x, name = None):
    if name is None: name = id_util.UniqueStr("Copy_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.copy_conf, "in", x.unique_name)
    op_conf.copy_conf.out = "out"
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return op_conf, lbi

def _MakeCopyInstructionBuilderFunction(x, op_conf):
    assert isinstance(x, remote_blob_util.EagerMirroredBlob), type(x)
    current_devices = oneflow.placement.current_scope().machine_id2device_id_list
    x_devices = x.blob_object.parallel_desc_symbol.machine_id2device_id_list
    assert current_devices == x_devices,\
            "\ncurrent_devices: %s\nx_devices: %s" %(current_devices, x_devices)
    current_device_tag = oneflow.placement.current_scope().default_device_tag
    x_device_tag = x.blob_object.parallel_desc_symbol.device_tag
    if current_device_tag == x_device_tag:
        return lambda builder: builder.DeprecatedStatelessCall(op_conf,
                const_arg_bns=["in"], mut_arg_bns=["out"])
    if current_device_tag == "cpu" and x_device_tag == "gpu":
        x_parallel_conf = x.blob_object.parallel_desc_symbol.parallel_conf
        return lambda builder: builder.DeprecatedCudaD2HStatelessCall(op_conf, x_parallel_conf,
                const_arg_bns=["in"], mut_arg_bns=["out"])
    if current_device_tag == "gpu" and x_device_tag == "cpu":
        out_parallel_conf = oneflow.placement.current_scope().default_parallel_conf
        def Build(builder):
            with builder.CudaHostPinBlob(x.blob_object):
                builder.DeprecatedCudaH2DStatelessCall(op_conf, out_parallel_conf,
                        const_arg_bns=["in"], mut_arg_bns=["out"])
        return Build
    raise NotImplementedError("invalid device found. current_device_tag: %s, x_device_tag: %s"
                              %(current_device_tag, x_device_tag))
