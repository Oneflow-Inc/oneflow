from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export


@oneflow_export("get_variable")
def get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    trainable=True,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
):
    r"""Create a new variable or get a existing variable by name.

    Args:
        name: name of this variable. Variable could be shared across different function created by annotation `@oneflow.function` :func:`~oneflow.function`. `None` by defauilt
        shape: shape of the variable. `None` by defauilt
        dtype: data type of the variable. `None` by defauilt
        initializer: a initializer_conf. For instance, a :func:`~oneflow.ones_initializer`. `None` by defauilt
        trainable: a `bool` to indicate if this variable is trainable. `True` by defauilt
        model_name: a `string`. `'weight'` or `'bias'`. `None` by defauilt
        random_seed: random seed for initialization. `None` by defauilt
    Returns:
        A `Blob`

    """
    assert isinstance(name, str)
    name = compile_context._get_variable_prefix() + name

    if name not in compile_context.cur_job_var_op_name2var_blob:
        op_conf = op_conf_util.OperatorConf()
        op_conf.name = name

        assert (
            shape is not None
        ), "Argument shape should not be None when the variable exists!"
        op_conf.variable_conf.shape.dim.extend(shape)

        if dtype is not None:
            op_conf.variable_conf.data_type = dtype
        if initializer is not None:
            op_conf.variable_conf.initializer.CopyFrom(initializer)
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

        compile_context.CurJobAddOp(op_conf)
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = op_conf.variable_conf.out
        var_blob = remote_blob_util.RemoteBlob(lbi)
        compile_context.cur_job_var_op_name2var_blob[name] = var_blob

    return compile_context.cur_job_var_op_name2var_blob[name]
