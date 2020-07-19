from __future__ import absolute_import

from contextlib import contextmanager

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.parallel_conf_util as parallel_conf_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.runtime_mode as runtime_mode
import oneflow.python.framework.push_util as push_util
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.lib.core.func_inspect_util as func_inspect_util
import oneflow.python.ops as ops

from oneflow.python.lib.core.box import Box
import oneflow


def Compile(session, function_desc, config_proto):
    with InterpretScope(session, function_desc, config_proto):
        _CompileJob(function_desc)
        c_api_util.CurJobBuildAndInferCtx_Complete()


def EagerRun(session, function_desc, config_proto, args):
    with InterpretScope(session, function_desc, config_proto):
        ret = _InterpretGlobalFunction(function_desc, args)
        c_api_util.CurJobBuildAndInferCtx_Complete()
    return ret


@contextmanager
def InterpretScope(session, function_desc, config_proto):
    job_conf = function_desc.job_config_proto
    job_conf.job_name = function_desc.job_func.__name__
    placement_scope = function_desc.function_attribute.default_placement_scope
    if placement_scope is None:
        tag_and_dev_ids = placement_util.GetDefaultMachineDeviceIds(
            oneflow.env.current_resource()
        )
        placement_scope = placement_util.GetPlacementScope(*tag_and_dev_ids)
    distribute_strategy = function_desc.function_attribute.default_distribute_strategy
    if distribute_strategy is None:
        distribute_strategy = distribute_util.DistributeMirroredStrategy()
    is_mirrored = isinstance(
        distribute_strategy, distribute_util.DistributeMirroredStrategy
    )
    tag_and_dev_ids = parallel_conf_util.GetDeviceTagAndMachineDeviceIds(
        placement_scope.default_parallel_conf
    )
    scope = _MakeInitialScope(job_conf, *tag_and_dev_ids, is_mirrored)
    with _JobBuildAndInferCtx(job_conf.job_name), placement_scope, distribute_strategy:
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)
        with runtime_mode.ModeScope(runtime_mode.GLOBAL_MODE):
            with _SessionInitialScope(session, scope):
                yield


def _SessionInitialScope(session, scope):
    job_name = scope.job_desc_symbol.data.job_name
    session.InitNoneScope(job_name)
    return session.NewCurrentScope(scope)


def _CompileJob(function_desc):
    func = function_desc.job_func
    func.__oneflow_input_blob_defs__ = _GetArgDefault(func)
    inputs = _RecursiveMakeInputBlobs(func.__oneflow_input_blob_defs__)
    kwarg = dict(
        allow_cpu_return_op=function_desc.function_attribute.allow_cpu_return_op
    )
    ret = func(*inputs)
    func.__oneflow_output_remote_blobs__ = _RecursiveMakeRetRemoteBlobs(ret, kwarg)


def _InterpretGlobalFunction(function_desc, args):
    func = function_desc.job_func
    func.__oneflow_input_blob_defs__ = _GetArgDefault(func)
    inputs = push_util.MakeEagerInputBlobs(func.__oneflow_input_blob_defs__, args)
    kwarg = dict(
        allow_cpu_return_op=function_desc.function_attribute.allow_cpu_return_op
    )
    ret = func(*inputs)
    return _RecursiveMakeRetRemoteBlobs(ret, kwarg)


@contextmanager
def _JobBuildAndInferCtx(job_name):
    c_api_util.JobBuildAndInferCtx_Open(job_name)
    try:
        yield
    finally:
        c_api_util.JobBuildAndInferCtx_Close()


def _GetArgDefault(func):
    if hasattr(func, "__oneflow_arg_default__"):
        return func.__oneflow_arg_default__
    return _CloneArgBlobDef(func_inspect_util.GetArgDefaults(func))


def _CloneArgBlobDef(args):
    if isinstance(args, input_blob_util.ArgBlobDef):
        return args.Clone()
    if isinstance(args, (tuple, list)):
        return type(args)(_CloneArgBlobDef(x) for x in args)
    if isinstance(args, dict):
        return {k: _CloneArgBlobDef(v) for k, v in args}
    raise NotImplementedError(
        "oneflow.global_function only accepts nested input blob defs"
    )


def _RecursiveMakeInputBlobs(input_blob_def):
    if isinstance(input_blob_def, input_blob_util.ArgBlobDef):
        return ops.InputOpByArgBlobDef(input_blob_def)
    if isinstance(input_blob_def, (tuple, list)):
        return type(input_blob_def)(_RecursiveMakeInputBlobs(x) for x in input_blob_def)
    if isinstance(input_blob_def, dict):
        return {k: _RecursiveMakeInputBlobs(v) for k, v in input_blob_def.items()}
    raise NotImplementedError(
        "oneflow.global_function accepts "
        + "ArgBlobDefs or list/tuple/dict nested ArgBlobDefs as argument"
    )


def _RecursiveMakeRetRemoteBlobs(remote_blobs, kwarg):
    if remote_blobs is None:
        return None
    if isinstance(remote_blobs, remote_blob_util.BlobDef):
        return ops.ReturnRemoteBlob(remote_blobs, **kwarg)
    if isinstance(remote_blobs, (tuple, list)):
        return type(remote_blobs)(
            _RecursiveMakeRetRemoteBlobs(x, kwarg) for x in remote_blobs
        )
    if isinstance(remote_blobs, dict):
        return {
            k: _RecursiveMakeRetRemoteBlobs(v, kwarg) for k, v in remote_blobs.items()
        }
    raise NotImplementedError(
        "oneflow.global_function returns "
        + "RemoteBlob or list/tuple/dict nested RemoteBlob only"
    )


def _MakeInitialScope(job_conf, device_tag, machine_device_ids, is_mirrored):
    scope = None

    def BuildInitialScope(builder):
        nonlocal scope
        scope = scope_util.BuildInitialScope(
            builder, job_conf, device_tag, machine_device_ids, is_mirrored
        )

    vm_util.LogicalRun(BuildInitialScope)
    return scope
