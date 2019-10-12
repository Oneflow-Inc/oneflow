def WatcherScopeStackPush(scope):
    global watcher_scope_stack
    watcher_scope_stack.insert(0, scope)

def WatcherScopeStackPop():
    global watcher_scope_stack
    watcher_scope_stack.pop(0)

def WatcherScopeStackTop():
    global watcher_scope_stack
    assert len(watcher_scope_stack) > 0, "no watcher scope found"
    return watcher_scope_stack[0]

watcher_scope_stack = []    
