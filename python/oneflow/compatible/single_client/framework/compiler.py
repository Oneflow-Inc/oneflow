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
import inspect
import typing
from contextlib import contextmanager

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import ops as ops
from oneflow.compatible.single_client.framework import c_api_util as c_api_util
from oneflow.compatible.single_client.framework import distribute as distribute_util
from oneflow.compatible.single_client.framework import hob as hob
from oneflow.compatible.single_client.framework import input_blob_def as input_blob_util
from oneflow.compatible.single_client.framework import (
    placement_context as placement_ctx,
)
from oneflow.compatible.single_client.framework import placement_util as placement_util
from oneflow.compatible.single_client.framework import push_util as push_util
from oneflow.compatible.single_client.framework import remote_blob as remote_blob_util
from oneflow.compatible.single_client.framework import runtime_mode as runtime_mode
from oneflow.compatible.single_client.framework import scope_util as scope_util
from oneflow.compatible.single_client.framework import session_context as session_ctx
from oneflow.compatible.single_client.framework import typing as oft
from oneflow.compatible.single_client.framework import typing_util as oft_util
from oneflow.compatible.single_client.support import enable_if as enable_if
from oneflow.compatible.single_client.support import (
    func_inspect_util as func_inspect_util,
)


def Compile(session, function_desc, config_proto):
    with InterpretScope(session, function_desc, config_proto):
        _CompileJob(session, function_desc)
        session.StashJob(function_desc.job_func.__name__)
        oneflow._oneflow_internal.CurJobBuildAndInferCtx_Complete()
        session.StashJob(
            function_desc.job_func.__name__,
            function_desc.job_func.__name__ + "_after_complete",
        )


def EagerRun(session, function_desc, config_proto, args):
    with InterpretScope(session, function_desc, config_proto):
        ret = _InterpretGlobalFunction(function_desc, args)
        oneflow._oneflow_internal.CurJobBuildAndInferCtx_Complete()
        session_ctx.GetDefaultSession().UpdateInfo4InterfaceOp()
    return ret


@contextmanager
def InterpretScope(session, function_desc, config_proto):
    job_conf = function_desc.job_config_proto
    job_conf.set_job_name(function_desc.job_func.__name__)
    placement_scope = function_desc.function_attribute.default_placement_scope
    if placement_scope is None:
        tag_and_dev_ids = placement_util.GetDefaultMachineDeviceIds(session.resource)
        hierarchy = None
    else:
        assert isinstance(placement_scope, placement_ctx.EmptyPlacementScope)
        tag_and_dev_ids = (
            placement_scope.device_tag,
            placement_scope.machine_device_ids,
        )
        hierarchy = placement_scope.hierarchy
    distribute_strategy = function_desc.function_attribute.default_distribute_strategy
    if distribute_strategy is None:
        distribute_strategy = distribute_util.DistributeConsistentStrategy()
    is_mirrored = isinstance(
        distribute_strategy, distribute_util.DistributeMirroredStrategy
    )
    assert isinstance(hierarchy, (list, tuple)) or hierarchy is None
    if hierarchy is not None:
        hierarchy = oneflow._oneflow_internal.Size(tuple(hierarchy))
    scope = scope_util.MakeInitialScope(
        job_conf, *tag_and_dev_ids, hierarchy, is_mirrored
    )
    with _JobBuildAndInferCtx(job_conf.job_name()), distribute_strategy:
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)
        with runtime_mode.ModeScope(runtime_mode.GLOBAL_MODE):
            with scope_util.ScopeContext(scope):
                yield


def _CompileJob(session, function_desc):
    func = function_desc.job_func
    parameters = func.__oneflow_function_signature__.parameters
    if len(parameters) == 0:
        func.__oneflow_input_blob_defs__ = ()
    elif all((p.annotation is inspect._empty for (_, p) in parameters.items())):
        func.__oneflow_input_blob_defs__ = _GetArgDefault(func)
    elif all((p.annotation is not inspect._empty for (_, p) in parameters.items())):
        func.__oneflow_input_blob_defs__ = _MakeInputBlobDefFromParameterSignature(
            parameters
        )
    else:
        raise NotImplementedError(
            "All parameters of global function should be annotated"
        )
    inputs = _RecursiveMakeInputBlobs(func.__oneflow_input_blob_defs__)
    ret = func(*inputs)
    return_annotation = func.__oneflow_function_signature__.return_annotation
    oft_util.CheckReturnByAnnotation(func.__name__, ret, return_annotation)
    func.__oneflow_output_remote_blobs__ = _RecursiveMakeRetRemoteBlobs(
        ret, allow_cpu_return_op=function_desc.function_attribute.allow_cpu_return_op
    )


def _InterpretGlobalFunction(function_desc, args):
    func = function_desc.job_func
    parameters = func.__oneflow_function_signature__.parameters
    if len(parameters) == 0:
        func.__oneflow_input_blob_defs__ = ()
    elif all((p.annotation is inspect._empty for (_, p) in parameters.items())):
        func.__oneflow_input_blob_defs__ = _GetArgDefault(func)
    elif all((p.annotation is not inspect._empty for (_, p) in parameters.items())):
        func.__oneflow_input_blob_defs__ = _MakeInputBlobDefFromParameterSignature(
            parameters
        )
    else:
        raise NotImplementedError(
            "All parameters of global function should be annotated"
        )
    inputs = push_util.MakeEagerInputBlobs(func.__oneflow_input_blob_defs__, args)
    ret = func(*inputs)
    return_annotation = func.__oneflow_function_signature__.return_annotation
    oft_util.CheckReturnByAnnotation(func.__name__, ret, return_annotation)
    return _RecursiveMakeRetRemoteBlobs(
        ret, allow_cpu_return_op=function_desc.function_attribute.allow_cpu_return_op
    )


@contextmanager
def _JobBuildAndInferCtx(job_name):
    c_api_util.JobBuildAndInferCtx_Open(job_name)
    try:
        yield
    finally:
        oneflow._oneflow_internal.JobBuildAndInferCtx_Close()


def _GetArgDefault(func):
    if hasattr(func, "__oneflow_arg_default__"):
        return func.__oneflow_arg_default__
    return _CloneArgBlobDef(func_inspect_util.GetArgDefaults(func))


def _CloneArgBlobDef(args):
    if isinstance(args, input_blob_util.ArgBlobDef):
        return args.Clone()
    if isinstance(args, (tuple, list)):
        return type(args)((_CloneArgBlobDef(x) for x in args))
    if isinstance(args, dict):
        return {k: _CloneArgBlobDef(v) for (k, v) in args}
    raise NotImplementedError(
        "oneflow.compatible.single_client.global_function only accepts nested input blob defs"
    )


def _RecursiveMakeInputBlobs(input_blob_def):
    if isinstance(input_blob_def, input_blob_util.ArgBlobDef):
        return ops.InputOpByArgBlobDef(input_blob_def)
    if isinstance(input_blob_def, (tuple, list)):
        return type(input_blob_def)(
            (_RecursiveMakeInputBlobs(x) for x in input_blob_def)
        )
    if isinstance(input_blob_def, dict):
        return {k: _RecursiveMakeInputBlobs(v) for (k, v) in input_blob_def.items()}
    raise NotImplementedError(
        "oneflow.compatible.single_client.global_function accepts "
        + "ArgBlobDefs or list/tuple/dict nested ArgBlobDefs as argument"
    )


def _MakeInputBlobDefFromParameterSignature(parameters):
    def CheckAndRecusiveMake(p):
        return _RecusiveMakeInputBlobDef(p.annotation)

    return tuple((CheckAndRecusiveMake(p) for (_, p) in parameters.items()))


def _RecusiveMakeInputBlobDef(cls):
    if oft.OriginFrom(cls, oft.OneflowNumpyDef):
        return cls.NewInputBlobDef()
    elif oft.OriginFrom(cls, typing.Tuple):
        return tuple((_RecusiveMakeInputBlobDef(a) for a in cls.__args__))
    else:
        raise NotImplementedError(
            "\nannotation %s" % cls
            + "not supported"
            + "\nonly support oneflow.compatible.single_client.typing.Numpy.Placeholder, oneflow.compatible.single_client.typing.ListNumpy.Placeholder"
        )


def _RecursiveMakeRetRemoteBlobs(remote_blobs, **kwarg):
    if remote_blobs is None:
        return None
    if isinstance(remote_blobs, oneflow._oneflow_internal.BlobDesc):
        return ops.ReturnRemoteBlob(remote_blobs, **kwarg)
    if isinstance(remote_blobs, (tuple, list)):
        return type(remote_blobs)(
            (_RecursiveMakeRetRemoteBlobs(x, **kwarg) for x in remote_blobs)
        )
    if isinstance(remote_blobs, dict):
        return {
            k: _RecursiveMakeRetRemoteBlobs(v, **kwarg)
            for (k, v) in remote_blobs.items()
        }
    raise NotImplementedError(
        "oneflow.compatible.single_client.global_function returns "
        + "RemoteBlob or list/tuple/dict nested RemoteBlob only"
    )
