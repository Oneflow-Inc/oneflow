from __future__ import absolute_import
import oneflow.python.framework.input_blob_def as input_blob_util

def AsyncPush(session, job_func, *arg):
    job_name = job_func.__name__
    assert len(arg) == len(job_func.__oneflow_input_blob_defs__)
    for i in range(len(arg)):
        _AsyncPushArg(session, job_func.__oneflow_input_blob_defs__[i], arg[i])

def _AsyncPushArg(session, arg_blob_def, arg_ndarray):
    if isinstance(arg_blob_def, (list, tuple)):
        assert type(arg_blob_def) is type(arg_ndarray)
        assert len(arg_blob_def) == len(arg_ndarray)
        for blob_def, ndarray in zip(arg_blob_def, arg_ndarray):
            _AsyncPushArg(session, blob_def, ndarray)
    elif isinstance(arg_blob_def, dict):
        assert type(arg_blob_def) is type(arg_ndarray)
        assert set(arg_blob_def.keys()) == set(arg_ndarray.keys())
        for k, blob_def in arg_blob_def.items():
            _AsyncPushArg(session, blob_def, arg_ndarray[k])
    else:
        assert isinstance(arg_blob_def, input_blob_util.ArgBlobDef)
        arg_blob_def.CheckAndAsyncPush(session, arg_ndarray)
