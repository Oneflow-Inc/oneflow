"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from typing import Optional, Sequence
from oneflow.python.oneflow_export import oneflow_export

import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.experimental.name_scope as name_scope
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.hob as hob
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.eager.op_executor as op_executor
import oneflow.python.lib.core.enable_if as enable_if
import oneflow
import os


@oneflow_export("get_variable")
def api_get_variable(
    name: str,
    shape: Optional[Sequence[int]] = None,
    dtype: Optional[dtype_util.dtype] = dtype_util.float32,
    initializer: Optional[op_conf_util.InitializerConf] = None,
    regularizer: Optional[op_conf_util.RegularizerConf] = None,
    trainable: Optional[bool] = None,
    model_name: Optional[str] = None,
    random_seed: Optional[int] = None,
    distribute: distribute_util.Distribute = distribute_util.broadcast(),
    reuse: bool = True,
) -> remote_blob_util.BlobDef:
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
    api = enable_if.unique([get_lazy_variable, get_eager_variable])
    return api(
        name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        model_name=model_name,
        random_seed=random_seed,
        distribute=distribute,
        reuse=reuse,
    )


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
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
    reuse=True,
):
    assert isinstance(name, str)
    assert isinstance(
        shape, (list, tuple)
    ), "param shape should be a list or tuple of dimension"

    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    name = name_scope.GetJobNameScopePrefix(job_name) + name
    sess = session_ctx.GetDefaultSession()
    var_blob, job_var_blob = sess.TryGetVariableBlobOfJobFromStash(job_name, name)

    if reuse is False:
        assert job_var_blob is None, (
            "varaible '{}' already exists, "
            "getting the same variable is not allowed "
            "when reuse is False".format(name)
        )

    if job_var_blob is None:
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
        op_attribute = compile_context.CurJobAddConsistentOp(op_conf)
        if var_blob is None:
            var_blob = _CreateEagerVariableBlob(op_attribute)
            op_executor.EagerInitVariableBlob(sess, op_conf, var_blob)

        assert isinstance(var_blob, remote_blob_util.EagerConsistentBlob)
        sess.StashVariableBlob4Job(job_name, op_conf.name, var_blob)
    else:
        assert isinstance(job_var_blob, remote_blob_util.EagerConsistentBlob)
        assert isinstance(var_blob, remote_blob_util.EagerConsistentBlob)
        assert var_blob.IdenticalTo(job_var_blob)

    bw_blob_register = gradient_util.GetDefaultBackwardBlobRegister()
    bw_blob_register.TrySetObject4BlobName(
        var_blob.logical_blob_name, var_blob.blob_object
    )
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
    reuse=True,
):
    assert isinstance(name, str)
    assert isinstance(
        shape, (list, tuple)
    ), "param shape should be a list or tuple of dimension"

    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    name = name_scope.GetJobNameScopePrefix(job_name) + name
    sess = session_ctx.GetDefaultSession()
    var_blob, job_var_blob = sess.TryGetVariableBlobOfJobFromStash(job_name, name)

    if reuse is False:
        assert job_var_blob is None, (
            "varaible '{}' already exists, "
            "getting the same variable is not allowed "
            "when param reuse is False".format(name)
        )

    if job_var_blob is None:
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
        job_var_blob, op_attr = _CreateVariableBlob(op_conf)
        assert isinstance(job_var_blob, remote_blob_util.LazyConsistentBlob)
        sess.AddVarOpAttr(op_conf.name, op_attr)
        sess.StashVariableBlob4Job(job_name, op_conf.name, job_var_blob)
        if var_blob is not None:
            assert isinstance(var_blob, remote_blob_util.LazyConsistentBlob)
            assert var_blob.IdenticalTo(job_var_blob)
    else:
        assert isinstance(job_var_blob, remote_blob_util.LazyConsistentBlob)
        assert isinstance(var_blob, remote_blob_util.LazyConsistentBlob)
        assert var_blob.IdenticalTo(job_var_blob)

    return job_var_blob


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

    assert dtype is not None
    op_conf.variable_conf.data_type = dtype.oneflow_proto_dtype

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


def _CreateVariableBlob(op_conf):
    op_attr = compile_context.CurJobAddConsistentOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.variable_conf.out
    return remote_blob_util.RemoteBlob(lbi), op_attr


def _CreateEagerVariableBlob(op_attribute):
    bn_in_op2blob_object = {}

    def BuildInstruction(builder):
        parallel_conf = oneflow.placement.current_scope().default_parallel_conf
        builder.StatelessCall(
            op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
        )

    vm_util.LogicalRun(BuildInstruction)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_attribute.op_conf.name
    lbi.blob_name = op_attribute.op_conf.variable_conf.out
    return remote_blob_util.EagerLogicalBlob(
        lbi, blob_object=bn_in_op2blob_object["out"]
    )
