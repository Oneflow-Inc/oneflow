from __future__ import absolute_import

import os

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.experimental.name_scope as name_scope
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.session_context as session_context
import oneflow.python.framework.id_util as id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("get_variable")
def get_variable(
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
    assert isinstance(name, str)
    assert isinstance(
        shape, (list, tuple)
    ), "param shape should be a list or tuple of dimension"

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

    root_path = (
        compile_context.GetCurJobConfigProto().default_initialize_with_snapshot_path
    )
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
    compile_context.CurJobAddConsistentOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.variable_conf.out
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("assign")
def assign(ref, value, dtype=None, name=None):
    if name is None:
        name = id_util.UniqueStr("Assign_")

    if os.getenv("ENABLE_USER_OP") == "True":
        op = (
            flow.consistent_user_op_builder(name)
            .Op("assign")
            .Input("ref", [ref])
            .Input("value", [value])
            .Build()
        )
        op.InferAndTryRun()
    else:
        op_conf = op_conf_util.OperatorConf()
        setattr(op_conf, "name", name)
        op_conf.assign_conf.ref = ref.unique_name
        op_conf.assign_conf.value = value.unique_name
        compile_context.CurJobAddConsistentOp(op_conf)
