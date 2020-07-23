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

import typing
import inspect
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.typing as oft
import oneflow.python.experimental.enable_typing_check as enable_typing_check


def CheckGlobalFunctionAnnotation(signature):
    parameters = signature.parameters
    if all(p.annotation is not inspect._empty for _, p in parameters.items()):
        for _, p in parameters.items():
            assert (
                p.kind == inspect._ParameterKind.POSITIONAL_OR_KEYWORD
            ), "no parameters like *args or **kwargs supported"
            CheckGlobalFunctionParamAnnotation(p.annotation)
    elif enable_typing_check.typing_check_enabled:
        for param_name, p in parameters.items():
            if p.annotaion is inspect._empty:
                raise NotImplementedError("parameter %s is not annotated" % param_name)
    else:
        # do nothing
        pass
    return_annotation = signature.return_annotation
    if return_annotation is not inspect._empty:
        CheckGlobalFunctionReturnAnnotation(return_annotation)
    elif enable_typing_check.typing_check_enabled:
        raise NotImplementedError("no return annotation found.")
    else:
        # do nothing
        pass


def CheckGlobalFunctionParamAnnotation(cls):
    if oft.OriginFrom(cls, typing.Tuple):
        assert cls.__args__ is not None, "T in typing.Tuple[T, ...] cannot be omitted"
        assert len(cls.__args__) > 0
        for cls_arg in cls.__args__:
            CheckGlobalFunctionParamAnnotation(cls_arg)
    elif oft.OriginFrom(cls, oft.OneflowNumpyDef):
        pass
    else:
        raise NotImplementedError("invalid parameter annotation %s found" % cls)


def CheckGlobalFunctionReturnAnnotation(cls):
    if cls is None:
        pass
    elif oft.OriginFrom(cls, oft.Callback):
        assert (
            cls.__args__ is not None
        ), "T in oneflow.typing.Callback[T] cannot be omitted"
        assert len(cls.__args__) == 1
        _CheckGlobalFunctionReturnAnnotation(cls.__args__[0])
    else:
        _CheckGlobalFunctionReturnAnnotation(cls)


def _CheckGlobalFunctionReturnAnnotation(cls):
    if oft.OriginFrom(cls, typing.Tuple):
        assert cls.__args__ is not None, "T in typing.Tuple[T, ...] cannot be omitted"
        assert len(cls.__args__) > 0
        for cls_arg in cls.__args__:
            _CheckGlobalFunctionReturnAnnotation(cls_arg)
    elif oft.OriginFrom(cls, oft.PyStructCompatibleToBlob):
        pass
    else:
        raise NotImplementedError("invalid return annotation %s found" % cls)


def CheckReturnByAnnotation(function_name, ret, annotation):
    if annotation is inspect._empty:
        return
    if annotation is None:
        error_str = (
            "%s does not matched return annotation %s of global_function %s."
            % (ret, annotation, function_name)
        )
        assert ret is None, error_str
    elif oft.OriginFrom(annotation, oft.Callback):
        _CheckReturnByAnnotation(function_name, ret, annotation.__args__[0])
    else:
        _CheckReturnByAnnotation(function_name, ret, annotation)


def _CheckReturnByAnnotation(function_name, ret, annotation):
    error_str = "%s does not matched return annotation %s of global_function %s." % (
        ret,
        annotation,
        function_name,
    )
    if oft.OriginFrom(annotation, typing.Tuple):
        assert type(ret) is tuple, error_str
        assert len(ret) == len(annotation.__args__), "%s length compare: %s v.s. %s" % (
            error_str,
            len(ret),
            len(annotation.__args__),
        )
        for ret_i, annotation_i in zip(ret, annotation.__args__):
            _CheckReturnByAnnotation(function_name, ret_i, annotation_i)
    elif oft.OriginFrom(annotation, oft.Numpy):
        assert isinstance(ret, remote_blob_util.BlobDef), "type(ret): %s" % type(ret)
        assert not ret.is_dynamic, (
            "only fixed shaped blob compatible to oneflow.typing.Numpy. "
            "you can change annotation to oneflow.typing.ListNumpy "
            "or oneflow.typing.ListListNumpy"
        )
    elif oft.OriginFrom(annotation, oft.ListNumpy):
        assert isinstance(ret, remote_blob_util.BlobDef), "type(ret): %s" % type(ret)
    elif oft.OriginFrom(annotation, oft.ListListNumpy):
        assert isinstance(ret, remote_blob_util.BlobDef), "type(ret): %s" % type(ret)
    else:
        raise NotImplementedError("invalid return annotation %s found" % annotation)


def TransformGlobalFunctionResult(future_blob, annotation):
    if annotation is inspect._empty:
        return future_blob
    elif annotation is None:
        assert future_blob is None
        return None
    elif oft.OriginFrom(annotation, oft.Callback):
        annotation = annotation.__args__[0]

        def Transform(f):
            return lambda x: f(TransformReturnedLocalBlob(x, annotation))

        return lambda f: future_blob.async_get(Transform(f))
    else:
        return TransformReturnedLocalBlob(future_blob.get(), annotation)


def TransformReturnedLocalBlob(local_blob, annotation):
    if oft.OriginFrom(annotation, typing.Tuple):
        assert type(local_blob) is tuple
        assert len(local_blob) == len(annotation.__args__)
        pairs = zip(local_blob, annotation.__args__)
        return tuple(TransformReturnedLocalBlob(*pair) for pair in pairs)
    elif oft.OriginFrom(annotation, oft.PyStructCompatibleToBlob):
        return TransformLocalBlob(local_blob, annotation)
    else:
        raise NotImplementedError(
            "invalid watch callback parameter annotation %s found" % annotation
        )


def CheckWatchCallbackParameterAnnotation(parameters):
    assert len(parameters) == 1, "watch callback should accept only one parameter"
    annotation = parameters[list(parameters.keys())[0]].annotation
    if annotation is inspect._empty:
        if enable_typing_check.typing_check_enabled:
            raise NotImplementedError("the watch callback's parameter is not annotated")
        return
    if not oft.OriginFrom(annotation, oft.PyStructCompatibleToBlob):
        raise NotImplementedError(
            ("invalid watch callback paremeter annotation %s found. " % annotation)
            + "candidate annotations: oneflow.typing.Numpy, oneflow.typing.ListNumpy, "
            "oneflow.typing.ListListNumpy"
        )


def CheckWatchedBlobByAnnotation(blob, annotation):
    if annotation is inspect._empty:
        return
    if oft.OriginFrom(annotation, oft.Numpy):
        assert not blob.is_dynamic, (
            "only fixed shaped blob compatible to oneflow.typing.Numpy. "
            "you can change annotation to oneflow.typing.ListNumpy "
            "or oneflow.typing.ListListNumpy"
        )
    elif oft.OriginFrom(annotation, oft.ListNumpy):
        pass
    elif oft.OriginFrom(annotation, oft.ListListNumpy):
        pass
    else:
        raise NotImplementedError(
            "invalid watch callback parameter annotation %s found" % annotation
        )


def TransformWatchedBlob(future_blob, handler):
    parameters = inspect.signature(handler).parameters
    annotation = parameters[list(parameters.keys())[0]].annotation
    if annotation is inspect._empty:
        return future_blob
    return TransformLocalBlob(future_blob, annotation)


def TransformLocalBlob(future_blob, annotation):
    if oft.OriginFrom(annotation, oft.Numpy):
        return future_blob.numpy()
    elif oft.OriginFrom(annotation, oft.ListNumpy):
        return future_blob.numpy_list()
    elif oft.OriginFrom(annotation, oft.ListListNumpy):
        return future_blob.numpy_lists()
    else:
        raise NotImplementedError(
            "invalid watch callback parameter annotation %s found" % annotation
        )
