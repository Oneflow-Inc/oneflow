from __future__ import absolute_import
import re
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.hob as hob
from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.eager.device_scope_stack as device_scope_stack
import oneflow
import traceback


@oneflow_export("placement.current_scope")
def api_current_placement_scope() -> placement_ctx.PlacementScope:
    """Get current placement scope object.

    Returns:
        placement_ctx.PlacementScope: PlacementScope object

    Examples:
    
        if "cpu" == flow.placement.current_scope().default_device_tag:
            print("ops shall run in the cpu mode only")
    
    """
    api = enable_if.unique(
        [global_mode_cur_placement_scope, normal_mode_cur_placement_scope]
    )
    return api()


@enable_if.condition(hob.in_global_mode & hob.in_placement_scope)
def global_mode_cur_placement_scope():
    return placement_ctx.PlacementScopeStackTop()


@enable_if.condition(hob.in_normal_mode)
def normal_mode_cur_placement_scope():
    return device_scope_stack.CurrentPlacement()


def api_fixed_placement(
    device_tag: str, machine_device_ids: str
) -> placement_ctx.FixedPlacementScope:
    return enable_if.unique([GetFixedPlacementScope])(device_tag, machine_device_ids)


@enable_if.condition(
    hob.in_global_mode
    | (hob.in_normal_mode & hob.env_initialized & ~hob.session_initialized)
)
def GetFixedPlacementScope(device_tag, machine_device_ids):
    return placement_ctx.FixedPlacementScope(device_tag, machine_device_ids)


@oneflow_export("device_prior_placement", "fixed_placement")
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
) -> placement_ctx.DevicePriorPlacementScope:
    """Create a scope. All ops within the scope will run on specified device that placed by  "device_tag" and "machine_device_ids".

    Args:
        device_tag (str): Device tag, "cpu" or "gpu" only
        machine_device_ids (str): String that specifies what device(s) to use in the format "<NODE INDEX (RANGE)>:<DEVICE INDEX (RANGE)>". For example, "0:0" means use the device 0 of machine 0, and "1:4-6" means use device 4, 5, 6 of machine 1.

    Returns:
        placement_ctx.DevicePriorPlacementScope:  Placement scope
    
    Example::
    
        with flow.fixed_placement("gpu", "0:0"):
            logits = lenet(images, train=False)
            loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
            flow.losses.add_loss(loss)
    
    """
    return enable_if.unique([GetDevicePriorPlacementScope])(
        device_tag, machine_device_ids
    )


@enable_if.condition(
    hob.in_global_mode
    | (hob.in_normal_mode & hob.env_initialized & ~hob.session_initialized)
)
def GetDevicePriorPlacementScope(device_tag, machine_device_ids):
    return placement_ctx.DevicePriorPlacementScope(device_tag, machine_device_ids)


def GetDefaultMachineDeviceIds(resource):
    if resource.HasField("gpu_device_num"):
        return "gpu", placement_ctx.GetGpuMachineDeviceIds(resource)
    elif resource.HasField("cpu_device_num"):
        return "cpu", placement_ctx.GetCpuMachineDeviceIds(resource)
    else:
        raise NotImplementedError
