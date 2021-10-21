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
from typing import Callable, List, Optional, Sequence, Tuple, Union

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework import hob as hob
from oneflow.compatible.single_client.framework import id_util as id_util
from oneflow.compatible.single_client.framework import interpret_util as interpret_util
from oneflow.compatible.single_client.framework import remote_blob as remote_blob_util
from oneflow.compatible.single_client.support import enable_if as enable_if
from oneflow.core.operator import op_conf_pb2 as op_conf_util
from oneflow.core.register import logical_blob_id_pb2 as logical_blob_id_util


def api_distribute_clone(
    x: oneflow._oneflow_internal.BlobDesc, name: Optional[str] = None
) -> Tuple[oneflow._oneflow_internal.BlobDesc]:
    func = enable_if.unique([distribute_clone])
    return func(x, name=name)


@enable_if.condition(hob.in_global_mode)
def distribute_clone(x, name=None):
    if name is None:
        name = id_util.UniqueStr("DistributeClone_")
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = name
    setattr(op_conf.distribute_clone_conf, "in", x.unique_name)
    parallel_size = flow.current_scope().device_parallel_desc_symbol.parallel_num
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


def api_distribute_add(
    xs: Sequence[oneflow._oneflow_internal.BlobDesc], name: Optional[str] = None
) -> oneflow._oneflow_internal.BlobDesc:
    func = enable_if.unique([distribute_add])
    return func(xs, name=name)


@enable_if.condition(hob.in_global_mode)
def distribute_add(xs, name=None):
    assert flow.current_scope().device_parallel_desc_symbol.parallel_num == len(xs)
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


def api_distribute_split(
    x: oneflow._oneflow_internal.BlobDesc, axis: int = 0, name: Optional[str] = None
) -> Tuple[oneflow._oneflow_internal.BlobDesc]:
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
    parallel_size = flow.current_scope().device_parallel_desc_symbol.parallel_num
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


def api_distribute_concat(
    xs: Sequence[oneflow._oneflow_internal.BlobDesc],
    axis: int = 0,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
    func = enable_if.unique([distribute_concat])
    return func(xs, axis=axis, name=name)


@enable_if.condition(hob.in_global_mode)
def distribute_concat(xs, axis=0, name=None):
    assert flow.current_scope().device_parallel_desc_symbol.parallel_num == len(xs)
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


def api_distribute_map(
    xs: Union[
        Sequence[oneflow._oneflow_internal.BlobDesc], oneflow._oneflow_internal.BlobDesc
    ],
    f: Callable[
        [oneflow._oneflow_internal.BlobDesc, oneflow._oneflow_internal.BlobDesc],
        oneflow._oneflow_internal.BlobDesc,
    ],
    axis: int = 0,
) -> Tuple[oneflow._oneflow_internal.BlobDesc]:
    func = enable_if.unqiue([distribute_map])
    return func(xs, f, axis=axis)


@enable_if.condition(hob.in_global_mode)
def distribute_map(xs, f, axis=0):
    _AssertInputOrOutput(xs)
    if isinstance(xs, (list, tuple)) == False:
        xs = [xs]
    splitted_xs = [flow.advanced.distribute_split(x, axis=axis) for x in xs]
    results = [_UnderSingleDevicePlacementScope(f, *x) for x in zip(*splitted_xs)]
    output_is_not_container = all(
        [isinstance(x, oneflow._oneflow_internal.ConsistentBlob) for x in results]
    )
    results = [_TryWrapTuple(x) for x in results]
    result = [flow.advanced.distribute_concat(x, axis=axis) for x in zip(*results)]
    if output_is_not_container:
        return result[0]
    return tuple(result)


def cast_to_current_logical_view(
    x: oneflow._oneflow_internal.BlobDesc,
) -> oneflow._oneflow_internal.BlobDesc:
    if (
        isinstance(x, oneflow._oneflow_internal.ConsistentBlob)
        and flow.scope.mirrored_view_enabled()
        or (
            isinstance(x, oneflow._oneflow_internal.MirroredBlob)
            and flow.scope.consistent_view_enabled()
        )
    ):
        x = flow.identity(x)
    return x


def _SoleConsistentLbn(blob):
    assert blob.parallel_size == 1
    if isinstance(blob, oneflow._oneflow_internal.ConsistentBlob):
        return blob.unique_name
    if isinstance(blob, oneflow._oneflow_internal.MirroredBlob):
        return blob.sub_consistent_blob_list[0].unique_name
    raise NotImplementedError


def _AssertInputOrOutput(xs):
    assert isinstance(xs, (list, tuple, oneflow._oneflow_internal.ConsistentBlob))
    if isinstance(xs, (list, tuple)):
        assert len(xs) > 0
        assert all(
            [isinstance(x, oneflow._oneflow_internal.ConsistentBlob) for x in xs]
        )


def _TryWrapTuple(ys):
    _AssertInputOrOutput(ys)
    if isinstance(ys, (list, tuple)) == False:
        ys = (ys,)
    return ys


def _UnderSingleDevicePlacementScope(f, *args):
    parallel_desc_symbol = flow.current_scope().device_parallel_desc_symbol
    for (machine_id, device_id) in _EachMachineIdAndDeviceId(parallel_desc_symbol):
        mch_dev_str = "@%d:%d" % (machine_id, device_id)
        with flow.scope.placement(parallel_desc_symbol.device_tag, mch_dev_str):
            return f(*args)


def _EachMachineIdAndDeviceId(parallel_desc_symbol):
    for (
        machine_id,
        device_id_list,
    ) in parallel_desc_symbol.machine_id2device_id_list.items():
        for device_id in device_id_list:
            yield (machine_id, device_id)
