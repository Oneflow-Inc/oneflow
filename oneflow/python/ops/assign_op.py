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

@oneflow_export("system.assign")
def api_assign(ref, value, validate_shape=None, use_locking=None, name=None):
    # TODO(lixinqi): check ref.is_lvalue
    return enable_if.unique(lazy_assign, eager_assign)(
            ref, value, validate_shape=validate_shape, use_locking=use_locking, name=name)

@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def lazy_assign(ref, value, validate_shape=None, use_locking=None, name=None):
    op_conf = _AssignOpConf(ref, value, name=name)
    compile_context.CurJobAddOp(op_conf, ref.parallel_conf)
    return ref

@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def eager_assign(ref, value, validate_shape=None, use_locking=None, name=None):
    op_conf = _AssignOpConf(ref, value, name=name)
    # no backward for assign
    vm_util.LogicalRun(vm_util.MakeFunctionAssignInstructionBuilder(
                ref.blob_object, value.blob_object, op_conf))
    return ref

def _AssignOpConf(ref, value, name = None):
    if name is None: name = id_util.UniqueStr("Assign_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    op_conf.assign_conf.ref = ref.unique_name
    op_conf.assign_conf.value = value.unique_name
    return op_conf
