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

import oneflow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.interpret_util as interpret_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
from typing import Union, Tuple, List, Optional, Sequence, Callable
import oneflow_api


@oneflow_export("advanced.distribute_clone")
def api_distribute_clone(
    x: oneflow_api.BlobDesc, name: Optional[str] = None
) -> Tuple[oneflow_api.BlobDesc]:
    func = enable_if.unique([distribute_clone])
    return func(x, name=name)


@enable_if.condition(hob.in_global_mode)
def distribute_clone(x, name=None):
    if name is None:
        name = id_util.UniqueStr("DistributeClone_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.distribute_clone_conf, "in", x.unique_name)
    parallel_size = oneflow.current_scope().device_parallel_desc_symbol.parallel_num
    op_conf.distribute_clone_conf.out.extend(
        ["out_%d" % i for i in range(parallel_size)]
    )
    interpret_util.ConsistentForward(op_conf)
    ret = []
    for i in range(parallel_size):
        out = "out_%d" % i
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = out
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return tuple(ret)


@oneflow_export("advanced.distribute_add")
def api_distribute_add(
    xs: Sequence[oneflow_api.BlobDesc], name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    func = enable_if.unique([distribute_add])
    return func(xs, name=name)


@enable_if.condition(hob.in_global_mode)
def distribute_add(xs, name=None):
    assert oneflow.current_scope().device_parallel_desc_symbol.parallel_num == len(xs)
    if name is None:
        name = id_util.UniqueStr("DistributeAdd_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    getattr(op_conf.distribute_add_conf, "in").extend(
        [_SoleConsistentLbn(x) for x in xs]
    )
    op_conf.distribute_add_conf.out = "out"
    interpret_util.ConsistentForward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("advanced.distribute_split")
def api_distribute_split(
    x: oneflow_api.BlobDesc, axis: int = 0, name: Optional[str] = None
) -> Tuple[oneflow_api.BlobDesc]:
    func = enable_if.unique([distribute_split])
    return func(x, axis=axis, name=name)


@enable_if.condition(hob.in_global_mode)
def distribute_split(x, axis=0, name=None):
    if name is None:
        name = id_util.UniqueStr("DistributeSplit_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.distribute_split_conf, "in", x.unique_name)
    op_conf.distribute_split_conf.axis = axis
    parallel_size = oneflow.current_scope().device_parallel_desc_symbol.parallel_num
    op_conf.distribute_split_conf.out.extend(
        ["out_%d" % i for i in range(parallel_size)]
    )
    interpret_util.ConsistentForward(op_conf)
    ret = []
    for i in range(parallel_size):
        out = "out_%d" % i
        lbi = logical_blob_id_util.LogicalBlobId()
        lbi.op_name = op_conf.name
        lbi.blob_name = out
        ret.append(remote_blob_util.RemoteBlob(lbi))
    return tuple(ret)


@oneflow_export("advanced.distribute_concat")
def api_distribute_concat(
    xs: Sequence[oneflow_api.BlobDesc], axis: int = 0, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    func = enable_if.unique([distribute_concat])
    return func(xs, axis=axis, name=name)


@enable_if.condition(hob.in_global_mode)
def distribute_concat(xs, axis=0, name=None):
    assert oneflow.current_scope().device_parallel_desc_symbol.parallel_num == len(xs)
    if name is None:
        name = id_util.UniqueStr("DistributeConcat_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    getattr(op_conf.distribute_concat_conf, "in").extend(
        [_SoleConsistentLbn(x) for x in xs]
    )
    op_conf.distribute_concat_conf.axis = axis
    op_conf.distribute_concat_conf.out = "out"
    interpret_util.ConsistentForward(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)


@oneflow_export("advanced.distribute_map")
def api_distribute_map(
    xs: Union[Sequence[oneflow_api.BlobDesc], oneflow_api.BlobDesc],
    f: Callable[[oneflow_api.BlobDesc, oneflow_api.BlobDesc], oneflow_api.BlobDesc],
    axis: int = 0,
) -> Tuple[oneflow_api.BlobDesc]:
    func = enable_if.unqiue([distribute_map])
    return func(xs, f, axis=axis)


@enable_if.condition(hob.in_global_mode)
def distribute_map(xs, f, axis=0):
    _AssertInputOrOutput(xs)
    if isinstance(xs, (list, tuple)) == False:
        xs = [xs]
    splitted_xs = [oneflow.advanced.distribute_split(x, axis=axis) for x in xs]
    results = [_UnderSingleDevicePlacementScope(f, *x) for x in zip(*splitted_xs)]
    output_is_not_container = all(
        [isinstance(x, oneflow_api.ConsistentBlob) for x in results]
    )
    results = [_TryWrapTuple(x) for x in results]
    result = [oneflow.advanced.distribute_concat(x, axis=axis) for x in zip(*results)]
    if output_is_not_container:
        return result[0]
    return tuple(result)


@oneflow_export("cast_to_current_logical_view")
def cast_to_current_logical_view(x: oneflow_api.BlobDesc,) -> oneflow_api.BlobDesc:
    if (
        isinstance(x, oneflow_api.ConsistentBlob)
        and oneflow.scope.mirrored_view_enabled()
    ) or (
        isinstance(x, oneflow_api.MirroredBlob)
        and oneflow.scope.consistent_view_enabled()
    ):
        x = oneflow.identity(x)
    return x


def _SoleConsistentLbn(blob):
    assert blob.parallel_size == 1
    if isinstance(blob, oneflow_api.ConsistentBlob):
        return blob.unique_name
    if isinstance(blob, oneflow_api.MirroredBlob):
        return blob.sub_consistent_blob_list[0].unique_name
    raise NotImplementedError


def _AssertInputOrOutput(xs):
    assert isinstance(xs, (list, tuple, oneflow_api.ConsistentBlob))
    if isinstance(xs, (list, tuple)):
        assert len(xs) > 0
        assert all([isinstance(x, oneflow_api.ConsistentBlob) for x in xs])


def _TryWrapTuple(ys):
    _AssertInputOrOutput(ys)
    if isinstance(ys, (list, tuple)) == False:
        ys = (ys,)
    return ys


def _UnderSingleDevicePlacementScope(f, *args):
    parallel_desc_symbol = oneflow.current_scope().device_parallel_desc_symbol
    for machine_id, device_id in _EachMachineIdAndDeviceId(parallel_desc_symbol):
        mch_dev_str = "%d:%d" % (machine_id, device_id)
        with oneflow.scope.placement(parallel_desc_symbol.device_tag, mch_dev_str):
            return f(*args)


def _EachMachineIdAndDeviceId(parallel_desc_symbol):
    for (
        machine_id,
        device_id_list,
    ) in parallel_desc_symbol.machine_id2device_id_list.items():
        for device_id in device_id_list:
            yield machine_id, device_id
