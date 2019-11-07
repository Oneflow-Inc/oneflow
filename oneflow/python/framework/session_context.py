import functools

def GetDefaultSession():
    assert _default_session is not None
    return _default_session

def InitDefaultSession(sess):
    global _default_session
    assert _default_session is None
    _default_session = sess

def TryDestroyDefaultSession():
    global _default_session
    assert _default_session is not None
    if _default_session is not None: _default_session.Destroy()
    _default_session = None
    
def try_init_default_session(func):
    @functools.wraps(func)
    def Func(*args):
        if GetDefaultSession().is_running == False : GetDefaultSession().Init()
        return func(*args)
    return Func

_default_session = None
