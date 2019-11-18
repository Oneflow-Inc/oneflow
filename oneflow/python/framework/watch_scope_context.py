def WatchScopeStackPush(scope):
    global watch_scope_stack
    watch_scope_stack.insert(0, scope)

def WatchScopeStackPop():
    global watch_scope_stack
    assert len(watch_scope_stack) > 0, "no watcher scope found"
    watch_scope_stack.pop(0)

def EachWatchScope():
    for watch_scope in watch_scope_stack: yield watch_scope

watch_scope_stack = []    
