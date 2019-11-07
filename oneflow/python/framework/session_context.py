import functools

def GetDefaultSession():
    assert _default_session is not None
    return _default_session

def ResetDefaultSession(sess):
    global _default_session
    _default_session = sess
    
def try_init_default_session(func):
    @functools.wraps(func)
    def Func(*args):
        if GetDefaultSession().is_running == False : GetDefaultSession().Init()
        return func(*args)
    return Func

_default_session = None
