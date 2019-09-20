def PlacementScopeStackPush(placement_policy):
    global placement_scope_stack
    placement_scope_stack.insert(0, placement_policy)

def PlacementScopeStackPop():
    global placement_scope_stack
    return placement_scope_stack.pop(0)
    
def CurPlacementGroupAddOpConf(op_conf):
    global placement_scope_stack
    assert len(placement_scope_stack) > 0
    placement_scope_stack[0].AppendOpConf(op_conf)

def CurPlacementGroupGetDeviceType(op_conf):
    global placement_scope_stack
    assert len(placement_scope_stack) > 0
    return placement_scope_stack[0].GetDeviceType4OpConf(op_conf)

def ParallelConf4OpConf(op_conf):
    global placement_scope_stack
    assert len(placement_scope_stack) > 0
    return placement_scope_stack[0].ParallelConf4OpConf(op_conf)


placement_scope_stack = []    
