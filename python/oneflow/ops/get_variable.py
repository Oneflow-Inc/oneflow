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
import os
from typing import Optional, Sequence, Union

import oneflow
import oneflow._oneflow_internal
import oneflow._oneflow_internal.oneflow.core.register.logical_blob_id as lbi_util
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.core.job.regularizer_conf_pb2 as regularizer_conf_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.eager.boxing_util as boxing_util
import oneflow.eager.gradient_util as gradient_util
import oneflow.eager.op_executor as op_executor
import oneflow.experimental.namescope as name_scope
import oneflow.framework.compile_context as compile_context
import oneflow.framework.distribute as distribute_util
import oneflow.framework.hob as hob
import oneflow.framework.remote_blob as remote_blob_util
import oneflow.framework.runtime_mode as rt_mode
import oneflow.framework.session_context as session_ctx
import oneflow.support.enable_if as enable_if

blob_register = oneflow._oneflow_internal.GetDefaultBlobRegister()


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
    parallel_distribution=None,
    reuse=True,
):
    assert isinstance(name, str)
    assert isinstance(
        shape, (list, tuple)
    ), "param shape should be a list or tuple of dimension"
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    name = name_scope.GetJobNameScopePrefix(job_name) + name
    sess = session_ctx.GetDefaultSession()
    (var_blob, job_var_blob) = sess.TryGetVariableBlobOfJobFromStash(job_name, name)
    if reuse is False:
        assert (
            job_var_blob is None
        ), "variable '{}' already exists, getting the same variable is not allowed when reuse is False".format(
            name
        )
    if job_var_blob is None:
        op_conf = GenerateVariableOpConf(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            trainable=trainable,
            model_name=model_name,
            random_seed=random_seed,
            parallel_distribution=parallel_distribution,
        )
        op_attribute = compile_context.CurJobAddConsistentOp(op_conf)
        if var_blob is None:
            var_blob = CreateEagerVariableBlob(op_attribute)
            op_executor.EagerInitVariableBlob(sess, op_conf, var_blob)
        assert isinstance(var_blob, oneflow._oneflow_internal.EagerConsistentBlob)
        sess.StashVariableBlob4Job(job_name, op_conf.name, var_blob)
    else:
        assert isinstance(job_var_blob, oneflow._oneflow_internal.EagerConsistentBlob)
        assert isinstance(var_blob, oneflow._oneflow_internal.EagerConsistentBlob)
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
    parallel_distribution=None,
    reuse=True,
):
    assert isinstance(name, str)
    assert isinstance(
        shape, (list, tuple)
    ), "param shape should be a list or tuple of dimension"
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    name = name_scope.GetJobNameScopePrefix(job_name) + name
    sess = session_ctx.GetDefaultSession()
    (var_blob, job_var_blob) = sess.TryGetVariableBlobOfJobFromStash(job_name, name)
    if reuse is False:
        assert (
            job_var_blob is None
        ), "variable '{}' already exists, getting the same variable is not allowed when param reuse is False".format(
            name
        )
    if job_var_blob is None:
        op_conf = GenerateVariableOpConf(
            name=name,
            shape=shape,
            dtype=dtype,
            initializer=initializer,
            regularizer=regularizer,
            trainable=trainable,
            model_name=model_name,
            random_seed=random_seed,
            parallel_distribution=parallel_distribution,
        )
        job_var_blob = _CreateVariableBlob(op_conf)
        assert isinstance(job_var_blob, oneflow._oneflow_internal.LazyConsistentBlob)
        sess.StashVariableBlob4Job(job_name, op_conf.name, job_var_blob)
        if var_blob is not None:
            assert isinstance(var_blob, oneflow._oneflow_internal.LazyConsistentBlob)
            assert var_blob.IdenticalTo(job_var_blob)
    else:
        assert isinstance(job_var_blob, oneflow._oneflow_internal.LazyConsistentBlob)
        assert isinstance(var_blob, oneflow._oneflow_internal.LazyConsistentBlob)
        assert var_blob.IdenticalTo(job_var_blob)
    return job_var_blob


def GenerateVariableOpConf(
    name,
    shape,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    parallel_distribution=None,
):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    op_conf.variable_conf.shape.dim.extend(shape)
    assert dtype is not None
    op_conf.variable_conf.data_type = oneflow._oneflow_internal.deprecated.GetProtoDtype4OfDtype(
        dtype
    )
    if rt_mode.CurrentMode() == rt_mode.NORMAL_MODE:
        root_path = None
    else:
        root_path = (
            compile_context.GetCurJobConfigProto().default_initialize_with_snapshot_path()
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
        op_conf.variable_conf.trainable = trainable
    if model_name is not None:
        op_conf.variable_conf.model_name = model_name
    if parallel_distribution is None:
        parallel_distribution = []
    op_conf.variable_conf.parallel_distribution.extend(parallel_distribution)
    if random_seed is not None:
        op_conf.variable_conf.random_seed = random_seed
    op_conf.variable_conf.out = "out"
    return op_conf


def _CreateVariableBlob(op_conf):
    compile_context.CurJobAddConsistentOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.variable_conf.out
    return remote_blob_util.RemoteBlob(lbi)


def CreateEagerVariableBlob(op_attribute, job_name=""):
    bn_in_op2blob_object = oneflow._oneflow_internal.deprecated.BnInOp2BlobObject()

    def BuildInstruction(builder):
        parallel_conf = (
            oneflow.current_scope().device_parallel_desc_symbol.parallel_conf
        )
        cfg_op_attribute = oneflow._oneflow_internal.deprecated.MakeOpAttributeByString(
            str(op_attribute)
        )
        builder.StatelessCall(
            cfg_op_attribute, parallel_conf, bn_in_op2blob_object, boxing_util.BoxingTo
        )

    oneflow._oneflow_internal.deprecated.LogicalRun(BuildInstruction)
    lbi = lbi_util.LogicalBlobId()
    lbi.set_op_name(op_attribute.op_conf.name)
    lbi.set_blob_name(op_attribute.op_conf.variable_conf.out)
    if not isinstance(lbi, lbi_util.LogicalBlobId):
        cfg_lbi = lbi_util.LogicalBlobId()
        cfg_lbi.set_op_name(lbi.op_name)
        cfg_lbi.set_blob_name(lbi.blob_name)
        lbi = cfg_lbi
    return oneflow._oneflow_internal.EagerConsistentBlob(
        lbi,
        blob_object=bn_in_op2blob_object["out"],
        blob_register=blob_register,
        job_name=job_name,
    )
