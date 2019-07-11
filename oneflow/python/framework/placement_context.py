job_name2default_parallel_conf = {}

parallel_conf_stack = []

def ParallelConfStackPush(parallel_conf):
    global parallel_conf_stack
    parallel_conf_stack.insert(0, self.parallel_conf_)

def ParallelConfStackPop():
    global parallel_conf_stack
    parallel_conf_stack.pop(0)
    
def ParallelConfStackTop():
    assert len(parallel_conf_stack) > 0
    return parallel_conf_stack[0]
