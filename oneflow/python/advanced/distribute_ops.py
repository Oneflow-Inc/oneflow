from __future__ import absolute_import
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util

from oneflow.python.oneflow_export import oneflow_export
import oneflow
    
@oneflow_export("advanced.distribute_clone")
def distribute_clone(x, name=None):
    if name is None: name = id_util.UniqueStr("DistributeClone_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.distribute_clone_conf, "in", x.logical_blob_name)
    parallel_size = oneflow.placement.current_scope().parallel_size
    op_conf.distribute_clone_conf.out.extend(["out_%d" % i for i in range(parallel_size)])
    compile_context.CurJobAddConsistentOp(op_conf)
    ret = []
    for i in range(parallel_size):
        out = "out_%d" % i
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = out
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return tuple(ret)

@oneflow_export("advanced.distribute_add")
def distribute_add(xs, name=None):
    assert oneflow.placement.current_scope().parallel_size == len(xs)
    if name is None: name = id_util.UniqueStr("DistributeAdd_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    getattr(op_conf.distribute_add_conf, "in").extend([_SoleConsistentLbn(x) for x in xs])
    op_conf.distribute_add_conf.out = "out"
    compile_context.CurJobAddConsistentOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("advanced.distribute_split")
def distribute_split(x, axis=0, name=None):
    if name is None: name = id_util.UniqueStr("DistributeSplit_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.distribute_split_conf, "in", x.logical_blob_name)
    op_conf.distribute_split_conf.axis = axis
    parallel_size = oneflow.placement.current_scope().parallel_size
    op_conf.distribute_split_conf.out.extend(["out_%d" % i for i in range(parallel_size)])
    compile_context.CurJobAddConsistentOp(op_conf)
    ret = []
    for i in range(parallel_size):
        out = "out_%d" % i
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = out
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return tuple(ret)

@oneflow_export("advanced.distribute_concat")
def distribute_concat(xs, axis=0, name=None):
    assert oneflow.placement.current_scope().parallel_size == len(xs)
    if name is None: name = id_util.UniqueStr("DistributeConcat_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    getattr(op_conf.distribute_concat_conf, "in").extend([_SoleConsistentLbn(x) for x in xs])
    op_conf.distribute_concat_conf.axis = axis
    op_conf.distribute_concat_conf.out = "out"
    compile_context.CurJobAddConsistentOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)

@oneflow_export("advanced.distribute_map")
def distribute_map(xs, f, axis = 0):
    _AssertInputOrOutput(xs)
    if isinstance(xs, (list, tuple)) == False: xs = [xs]
    splitted_xs = [oneflow.advanced.distribute_split(x, axis=axis) for x in xs]
    results = [_UnderSingleDevicePlacementScope(f, *x) for x in zip(*splitted_xs)]
    output_is_not_container = all([isinstance(x, remote_blob_util.ConsistentBlob) for x in results])
    results = [_TryWrapTuple(x) for x in results]
    result = [oneflow.advanced.distribute_concat(x, axis=axis) for x in zip(*results)]
    if output_is_not_container: return result[0]
    return tuple(result)

def _SoleConsistentLbn(blob):
    assert blob.parallel_size == 1
    if isinstance(blob, remote_blob_util.ConsistentBlob): return blob.logical_blob_name
    if isinstance(blob, remote_blob_util.MirroredBlob):
      return blob.sub_consistent_blob_list[0].logical_blob_name
    raise NotImplementedError

def _AssertInputOrOutput(xs):
    assert isinstance(xs, (list, tuple, remote_blob_util.ConsistentBlob))
    if isinstance(xs, (list, tuple)):
        assert len(xs) > 0
        assert all([isinstance(x, remote_blob_util.ConsistentBlob) for x in xs])

def _TryWrapTuple(ys):
    _AssertInputOrOutput(ys)
    if isinstance(ys, (list, tuple)) == False: ys = (ys,)
    return ys

def _UnderSingleDevicePlacementScope(f, *args):
    current_scope = oneflow.placement.current_scope()
    for machine_id, device_id in _EachMachineIdAndDeviceId(current_scope):
        mch_dev_str = "%d:%d" % (machine_id, device_id)
        with oneflow.device_prior_placement(current_scope.default_device_tag, mch_dev_str):
            return f(*args)

def _EachMachineIdAndDeviceId(placement_scope):
    for machine_id, device_id_list in placement_scope.machine_id2device_id_list.items():
        for device_id in device_id_list: yield machine_id, device_id
