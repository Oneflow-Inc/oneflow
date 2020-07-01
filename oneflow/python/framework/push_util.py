from __future__ import absolute_import

import oneflow
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.python_callback as python_callback
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.python.eager.object as object_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import numpy


def AsyncPush(session, job_func, *arg):
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


def MakeEagerInputBlobs(arg_blob_def_tup, arg_ndarray_tup):
    assert isinstance(arg_blob_def_tup, (list, tuple))
    assert isinstance(arg_ndarray_tup, (list, tuple))
    assert len(arg_blob_def_tup) == len(arg_ndarray_tup)

    return type(arg_blob_def_tup)(
        _CreateEagerInputBlobAndFeedValue(arg_blob_def, arg_ndarray)
        for arg_blob_def, arg_ndarray in zip(arg_blob_def_tup, arg_ndarray_tup)
    )


def _CheckInputArgBlobDefValueMatch(arg_blob_def, arg_value):
    if isinstance(arg_blob_def, input_blob_util.FixedTensorDef):
        assert isinstance(arg_value, numpy.ndarray)
        assert arg_blob_def.shape == arg_value.shape
    elif isinstance(arg_blob_def, input_blob_util.MirroredTensorDef):
        assert isinstance(arg_value, (list, tuple))
        for v in arg_value:
            assert isinstance(v, numpy.ndarray)
            assert len(v.shape) == len(arg_blob_def.shape)
            assert numpy.prod(v) <= numpy.prod(arg_blob_def.shape)
    elif isinstance(arg_blob_def, input_blob_util.MirroredTensorListDef):
        raise NotImplementedError
    else:
        raise NotImplementedError


def _MakeReleaser4InputBlobObject(lbi, rank):
    blob_register = blob_register_util.GetDefaultBlobRegister()
    lbn = "{}/{}/{}".format(lbi.op_name, lbi.blob_name, rank)

    def ReleaseInputBlobObject(*args):
        blob_register.ClearObject4BlobName(lbn)

    return ReleaseInputBlobObject


def _CreateEagerInputBlobAndFeedValue(arg_blob_def, arg_ndarray):
    _CheckInputArgBlobDefValueMatch(arg_blob_def, arg_ndarray)
    arg_blob_object, lbi = _MakeInputBlobObject(arg_blob_def)
    physical_blob_objects = _GetPhysicalBlobObjects(arg_blob_object, lbi)
    for i, physical_blob_object in enumerate(physical_blob_objects):
        arg_blob_object.add_releaser(_MakeReleaser4InputBlobObject(lbi, i))
        _FeedValueToInputPhysicalBlob(
            arg_blob_def, physical_blob_object, arg_ndarray, i
        )
    return remote_blob_util.RemoteBlob(lbi, blob_object=arg_blob_object)


def _MakeInputBlobObject(arg_blob_def):
    input_op_conf, lbi = _MakeInputOpConfAndRetLbi(arg_blob_def)
    bn_in_op2blob_object = {}

    def BuildInputInstruction(builder):
        op_attribute = compile_context.CurJobAddOp(input_op_conf)
        parallel_conf = oneflow.placement.current_scope().default_parallel_conf
        builder.StatelessCall(
            op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
        )

    vm_util.LogicalRun(BuildInputInstruction)
    return bn_in_op2blob_object["out"], lbi


def _GetPhysicalBlobObjects(logical_blob_object, lbi):
    blob_register = blob_register_util.GetDefaultBlobRegister()
    physical_blob_names = []

    def BuildLogical2PhysicalInstruction(builder):
        physical_blob_objects = builder.UnpackLogicalBlobToPhysicalBlobs(
            logical_blob_object
        )
        for i, physical_blob_object in enumerate(physical_blob_objects):
            blob_name = "{}/{}/{}".format(lbi.op_name, lbi.blob_name, i)
            physical_blob_names.append(blob_name)
            if not blob_register.HasObject4BlobName(blob_name):
                blob_register.SetObject4BlobName(blob_name, physical_blob_object)

    vm_util.LogicalRun(BuildLogical2PhysicalInstruction)
    return [blob_register.GetObject4BlobName(name) for name in physical_blob_names]


def _MakeInputOpConfAndRetLbi(arg_blob_def):
    assert isinstance(arg_blob_def, input_blob_util.ArgBlobDef)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("Input_")
    op_conf.input_conf.out = "out"
    op_conf.input_conf.blob_conf.CopyFrom(arg_blob_def.ToInterfaceBlobConf())
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.input_conf.out
    return op_conf, lbi


def _FeedValueToInputPhysicalBlob(blob_def, blob_object, arg_ndarray, rank):
    assert isinstance(blob_def, input_blob_util.ArgBlobDef)
    assert isinstance(blob_object, object_util.BlobObject)

    if isinstance(blob_def, input_blob_util.FixedTensorDef):
        assert isinstance(arg_ndarray, numpy.ndarray)

        def FeedBody(ofblob):
            assert ofblob.shape == arg_ndarray.shape
            if ofblob.CopyFromNdarray(arg_ndarray) is False:
                raise ValueError

        def BuildFeedInstruction(builder):
            builder.WatchBlobBody(blob_object, FeedBody)

        vm_util.PhysicalRun(BuildFeedInstruction)
        python_callback.DeleteRegisteredCallback(FeedBody)

    elif isinstance(blob_def, input_blob_util.MirroredTensorDef):
        assert isinstance(arg_ndarray, (list, tuple))
        op_arg_parallel_num = (
            blob_object.op_arg_parallel_attr.parallel_desc_symbol.parallel_num
        )
        assert len(arg_ndarray) == 1 or len(arg_ndarray) == op_arg_parallel_num
        assert all(isinstance(a, numpy.ndarray) for a in arg_ndarray)
        assert rank >= 0 and rank < op_arg_parallel_num

        def FeedHeader(ofblob):
            ndarray = arg_ndarray[0] if len(arg_ndarray) == 1 else arg_ndarray[rank]
            assert isinstance(ndarray, numpy.ndarray)
            ofblob.set_shape(ndarray.shape)

        def FeedBody(ofblob):
            value = arg_ndarray[0] if len(arg_ndarray) == 1 else arg_ndarray[rank]
            assert isinstance(value, numpy.ndarray)
            if ofblob.CopyFromNdarray(value) is False:
                raise ValueError

        def BuildFeedInstruction(builder):
            builder.WatchBlobHeader(blob_object, FeedHeader)
            builder.WatchBlobBody(blob_object, FeedBody)

        vm_util.PhysicalRun(BuildFeedInstruction)
        python_callback.DeleteRegisteredCallback(FeedHeader)
        python_callback.DeleteRegisteredCallback(FeedBody)

    elif isinstance(blob_def, input_blob_util.MirroredTensorListDef):
        raise NotImplementedError
    else:
        raise NotImplementedError
