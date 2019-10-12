from __future__ import absolute_import
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow

from oneflow.python.oneflow_export import oneflow_export

@oneflow_export("advanced.distribute_map")
def distribute_map(xs, f, axis = 0):
    _AssertInputOrOutput(xs)
    if isinstance(xs, (list, tuple)) == False: xs = [xs]
    splitted_xs = [oneflow.advanced.distribute_split(x, axis=axis) for x in xs]
    results = [_UnderSingleDevicePlacementScope(f, *x) for x in zip(*splitted_xs)]
    output_is_not_container = all([isinstance(x, remote_blob_util.RemoteBlob) for x in results])
    results = [_TryWrapTuple(x) for x in results]
    result = [oneflow.advanced.distribute_concat(x, axis=axis) for x in zip(*results)]
    if output_is_not_container: return result[0]
    return tuple(result)

def _AssertInputOrOutput(xs):
    assert isinstance(xs, (list, tuple, remote_blob_util.RemoteBlob))
    if isinstance(xs, (list, tuple)):
        assert len(xs) > 0
        assert all([isinstance(x, remote_blob_util.RemoteBlob) for x in xs])

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
