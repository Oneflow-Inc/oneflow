job_name2default_parallel_conf = {}

parallel_conf_stack = []

cur_placement_scope_op_names = None

def ParallelConfStackPush(parallel_conf):
    global parallel_conf_stack
    parallel_conf_stack.insert(0, parallel_conf)

def ParallelConfStackPop():
    global parallel_conf_stack
    parallel_conf_stack.pop(0)
    
def ParallelConfStackTop():
    assert len(parallel_conf_stack) > 0
    return parallel_conf_stack[0]

def CurPlacementGroupAddOpName(op_name):
    global cur_placement_scope_op_names
    cur_placement_scope_op_names.append(op_name)
