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
import re
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.framework.hob as hob
from oneflow.python.oneflow_export import oneflow_export, oneflow_deprecate
import oneflow.python.lib.core.enable_if as enable_if
import oneflow
import traceback
import oneflow_api


@oneflow_export("device_prior_placement", "fixed_placement")
@oneflow_deprecate()
def deprecated_placement(*args, **kwargs):
    print(
        "WARNING:",
        "oneflow.device_prior_placement/oneflow.fixed_placement",
        "will be removed in the future, use {} instead.".format(
            "oneflow.scope.placement"
        ),
    )
    print(traceback.format_stack()[-2])
    return api_placement(*args, **kwargs)


@oneflow_export("scope.placement")
def api_placement(
    device_tag: str, machine_device_ids: str
) -> placement_ctx.PlacementScope:
    r"""Create a scope. All ops within the scope will run on specified device that placed by  "device_tag" and "machine_device_ids".

    Args:
        device_tag (str): Device tag, "cpu" or "gpu" only
        machine_device_ids (str): List of string that specifies what machine & device(s) to use, the format is "List[<NODE INDEX>:<DEVICE START INDEX>-<DEVICE END INDEX>, <NODE INDEX>:<DEVICE START INDEX>-<DEVICE END INDEX>, ...]", For example, "0:0" means use the device 0 of machine 0, and "1:4-6" means use device 4, 5, 6 of machine 1.

    Returns:
        placement_ctx.DevicePriorPlacementScope:  Placement scope

    For example:

    If you run program on single machine, you can assign the specified device like this:

    .. code-block:: python

        with flow.scope.placement("gpu", "0:0"):
            logits = lenet(images, train=False)
            loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
            flow.losses.add_loss(loss)

    Or you run distributed program, you can assign the specified devices like this:

    .. code-block:: python

        # configure machines ids, ips, etc.
        with flow.scope.placement("gpu", ['0:0-7', '1:0-7']):
            logits = lenet(images, train=False)
            loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
            flow.losses.add_loss(loss)

    """

    if oneflow_api.flags.with_cuda() == False:
        device_tag = "cpu"
    func = enable_if.unique(
        [
            GetEmptyPlacementScope,
            GetNormalModePlacementScope,
            GetGlobalModePlacementScope,
        ]
    )
    return func(device_tag, machine_device_ids)


@enable_if.condition(
    hob.in_normal_mode & hob.env_initialized & ~hob.session_initialized
)
def GetEmptyPlacementScope(device_tag, machine_device_ids):
    return placement_ctx.EmptyPlacementScope(device_tag, machine_device_ids)


@enable_if.condition(hob.in_normal_mode & hob.session_initialized)
def GetNormalModePlacementScope(device_tag, machine_device_ids, hierarchy=None):
    if isinstance(machine_device_ids, tuple):
        machine_device_ids = list(machine_device_ids)
    if not isinstance(machine_device_ids, list):
        machine_device_ids = [machine_device_ids]
    sess = session_ctx.GetDefaultSession()
    assert isinstance(hierarchy, (list, tuple)) or hierarchy is None
    if hierarchy is not None:
        if type(hierarchy) is list:
            hierarchy = tuple(hierarchy)
        hierarchy = oneflow_api.Size(hierarchy)

    scope = scope_util.MakeScope(
        lambda old_scope, builder: builder.BuildScopeWithNewParallelDesc(
            old_scope, device_tag, machine_device_ids, hierarchy
        )
    )
    return scope_util.ScopeContext(scope)


@enable_if.condition(hob.in_global_mode)
def GetGlobalModePlacementScope(device_tag, machine_device_ids, hierarchy=None):
    if isinstance(machine_device_ids, (list, tuple)) == False:
        machine_device_ids = [machine_device_ids]
    sess = session_ctx.GetDefaultSession()
    assert isinstance(hierarchy, (list, tuple)) or hierarchy is None
    if hierarchy is not None:
        if type(hierarchy) is list:
            hierarchy = tuple(hierarchy)
        hierarchy = oneflow_api.Size(hierarchy)

    def BuildScope(old_scope, builder):
        return builder.BuildScopeWithNewParallelDesc(
            old_scope, device_tag, machine_device_ids, hierarchy
        )

    scope_ctx = scope_util.ScopeContext(scope_util.MakeScope(BuildScope))
    return placement_ctx.GlobalModePlacementScope(scope_ctx)


def GetDefaultMachineDeviceIds(resource):
    if resource.HasField("gpu_device_num") and resource.gpu_device_num > 0:
        return "gpu", placement_ctx.GetGpuMachineDeviceIds(resource)
    elif resource.HasField("cpu_device_num"):
        return "cpu", placement_ctx.GetCpuMachineDeviceIds(resource)
    else:
        raise NotImplementedError
