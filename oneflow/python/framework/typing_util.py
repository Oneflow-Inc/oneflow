from __future__ import absolute_import

import typing
import inspect
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.typing as oft


def CheckGlobalFunctionAnnotation(signature):
    parameters = signature.parameters
    if all(p.annotation is not inspect._empty for _, p in parameters.items()):
        for _, p in parameters.items():
            assert (
                p.kind == inspect._ParameterKind.POSITIONAL_OR_KEYWORD
            ), "no parameters like *args or **kwargs supported"
            CheckGlobalFunctionParamAnnotation(p.annotation)
    return_annotation = signature.return_annotation
    if return_annotation is not inspect._empty:
        CheckGlobalFunctionReturnAnnotation(return_annotation)


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


def CheckWatchCallbackParameterAnnotation(parameters):
    assert len(parameters) == 1, "watch callback should accept only one parameter"
    annotation = parameters[list(parameters.keys())[0]].annotation
    if annotation is inspect._empty:
        return
    if not oft.OriginFrom(annotation, oft.PyStructCompatibleToBlob):
        raise NotImplementedError(
            "invalid watch callback paremeter annotation. "
            "candidate annotations: oneflow.typing.Numpy, oneflow.typing.ListNumpy, "
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
