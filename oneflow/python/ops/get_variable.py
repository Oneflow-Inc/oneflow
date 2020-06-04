from __future__ import absolute_import

import oneflow.python.framework.session_context as session_context
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.experimental.name_scope as name_scope
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.ops.user_op_builder as user_op_builder_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.boxing_util as boxing_util
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
import oneflow

import os

@oneflow_export("get_variable")
def api_get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
):
    r"""Create a variable or retrieve an existing one.

    Args:
        name: Name of this variable. One variable could be shared by multiple OneFlow functions. `None` by defauilt
        shape: Shape of the variable. `None` by defauilt
        dtype: Data type of the variable. `None` by defauilt
        initializer: A initializer object. For instance, a :func:`~oneflow.ones_initializer`. `None` by defauilt
        trainable: A `bool` to indicate if this variable is trainable. `True` by defauilt
        model_name: A `string`. `'weight'` or `'bias'`. `None` by defauilt
        random_seed: Random seed for random initializers. `None` by defauilt

    """
    return enable_if.unique(get_lazy_variable, get_eager_variable)(
            name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            trainable=trainable,
            model_name=model_name,
            random_seed=random_seed,
            distribute=distribute)

@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled
                     & ~hob.consistent_view_enabled)
def get_eager_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
):
    assert isinstance(name, str)
    assert isinstance(shape, (list, tuple)), "param shape should be a list or tuple of dimension"
    # TODO(lixinqi) only BroadcastDistribute supported yet
    assert isinstance(distribute, distribute_util.BroadcastDistribute)

    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    name = name_scope.GetJobNameScopePrefix(job_name) + name
    sess = session_context.GetDefaultSession()
    var_blob = sess.TryGetVariableBlobOfJobFromStash(job_name, name)

    if var_blob is None:
        op_conf = _GenerateVariableOpConf(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            trainable=trainable,
            model_name=model_name,
            random_seed=random_seed,
            distribute=distribute,
        )
        op_conf, parallel_conf = compile_context.GetOpConfAndParallelConf(op_conf)
        var_blob = _CreateEagerVariableBlob(op_conf, parallel_conf)
        _InitVariableBlob(op_conf, var_blob)
        sess.StashVariableBlob4Job(job_name, op_conf.name, var_blob)
    assert var_blob.shape == shape
    assert var_blob.dtype == dtype
    return var_blob


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def get_lazy_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
):
    assert isinstance(name, str)
    assert isinstance(shape, (list, tuple)), "param shape should be a list or tuple of dimension"

    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    name = name_scope.GetJobNameScopePrefix(job_name) + name
    sess = session_context.GetDefaultSession()
    var_blob = sess.TryGetVariableBlobOfJobFromStash(job_name, name)

    if var_blob is not None:
        assert var_blob.shape == shape
        assert var_blob.dtype == dtype
    else:
        op_conf = _GenerateVariableOpConf(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            trainable=trainable,
            model_name=model_name,
            random_seed=random_seed,
            distribute=distribute,
        )
        op_conf, parallel_conf = compile_context.GetOpConfAndParallelConf(op_conf)
        var_blob = _CreateVariableBlob(op_conf, parallel_conf)
        sess.StashVariableBlob4Job(job_name, op_conf.name, var_blob)

    return var_blob


def _GenerateVariableOpConf(
    name,
    shape,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    op_conf.variable_conf.shape.dim.extend(shape)

    if dtype is not None:
        op_conf.variable_conf.data_type = dtype

    root_path = compile_context.GetCurJobConfigProto().default_initialize_with_snapshot_path
    dir_path = os.path.join(root_path, name)
    file_path = os.path.join(dir_path, "out")
    if root_path and os.path.isfile(file_path):
        op_conf.variable_conf.initialize_with_snapshot.path = dir_path
        op_conf.variable_conf.initialize_with_snapshot.key = "out"
    else:
        if root_path:
            print("{} not found, will be initialized".format(file_path))
        if initializer is not None:
            op_conf.variable_conf.initializer.CopyFrom(initializer)

    if regularizer is not None:
        op_conf.variable_conf.regularizer.CopyFrom(regularizer)

    if trainable is not None:
        op_conf.trainable = trainable

    if model_name is not None:
        op_conf.variable_conf.model_name = model_name

    if type(distribute) is distribute_util.SplitDistribute:
        op_conf.variable_conf.split_axis.value = distribute.axis
    else:
        op_conf.variable_conf.split_axis.ClearField("value")

    if random_seed is not None:
        op_conf.variable_conf.random_seed = random_seed

    op_conf.variable_conf.out = "out"
    return op_conf


def _CreateVariableBlob(op_conf, parallel_conf):
    compile_context.CurJobAddConsistentOp(op_conf, parallel_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.variable_conf.out
    return remote_blob_util.RemoteBlob(lbi)

def _CreateEagerVariableBlob(op_conf, parallel_conf):
    compile_context.CurJobAddMirroredOp(op_conf, parallel_conf)
    bn_in_op2blob_object = {}
    vm_util.LogicalRun(
            lambda builder: builder.SystemStatelessCall(op_conf, mut_arg_bns=['out'],
                bn_in_op2blob_object=bn_in_op2blob_object))
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.variable_conf.out
    return remote_blob_util.EagerLogicalBlob(lbi, blob_object=bn_in_op2blob_object['out'])

def _InitVariableBlob(var_op_conf, var_blob):
    with oneflow.fixed_placement("cpu", "0:0"):
        _Assign(var_blob.blob_object, _ModelInit(var_op_conf))
        
def _Assign(var_blob_object, value_blob_object):
    vm_util.LogicalRun(lambda builder: boxing_util.Assign(builder, var_blob_object, value_blob_object))

def _ModelInit(var_op_conf):
    op_conf, lbi = _GetModelInitAndLbi(var_op_conf)
    bn_in_op2blob_object = {}
    def BuildModeInitInstruction(builder):
        builder.SystemStatelessCall(op_conf, mut_arg_bns=['out_0'],
                bn_in_op2blob_object=bn_in_op2blob_object)
    vm_util.LogicalRun(BuildModeInitInstruction)
    return bn_in_op2blob_object['out_0']

def _GetModelInitAndLbi(var_op_conf):
    variable_op_conf = op_conf_util.VariableOpConf()
    variable_op_conf.CopyFrom(var_op_conf.variable_conf)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("ModelInit_")
    op_conf.model_init_conf.tick = "undefined-source_tick/out"
    op_conf.model_init_conf.out.append("out_0")
    op_conf.model_init_conf.variable_op_name.append(var_op_conf.name)
    op_conf.model_init_conf.original_variable_conf.append(variable_op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.model_init_conf.out[0]
    return op_conf, lbi
