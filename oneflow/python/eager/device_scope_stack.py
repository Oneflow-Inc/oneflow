import oneflow.python.lib.core.lazy as lazy
import oneflow.python.lib.core.scope_stack as scope_stack
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.c_api_util as c_api_util
import oneflow

class PhysicalPlacementScope(placement_ctx.FixedPlacementScope):
    def __init__(self, device_tag, machine_device_ids):
        placement_ctx.FixedPlacementScope.__init__(self, device_tag, machine_device_ids)
        self.physical_symbol_id_ = c_api_util.NewPhysicalSymbolId()

    @property
    def is_physical_placement(self): return True

    @property
    def physical_symbol_id(self): return self.physical_symbol_id_

def CurrentPlacement():
    global device_scope_stack
    return device_scope_stack.Current().value

def _GetInitDeviceScope():
    resource = oneflow.env.current_resource()
    if resource.HasField('gpu_device_num'):
        device_tag = 'gpu'
    elif resource.HasField('cpu_device_num'):
        device_tag = 'cpu'
    else:
        raise NotImplementedError
    return _NewPhysicalPlacementScope(device_tag, 0)

def _NewPhysicalPlacementScope(device_tag, device_id):
    current_machine_id = oneflow.current_machine_id()
    return PhysicalPlacementScope(device_tag, "%d:%s" % (current_machine_id, device_id))

device_scope_stack = scope_stack.ScopeStack(lazy.Lazy(_GetInitDeviceScope))
