import functools

class SessionStatus:
    OPEN = "OPEN"
    
    RUNNING = "RUNNING"

    CLOSED = "CLOSED"

def GetDefaultSession():
    assert _default_session is not None
    return _default_session

def OpenDefaultSession(sess):
    global _default_session
    assert _default_session is None
    _default_session = sess

def TryCloseDefaultSession():
    global _default_session
    assert _default_session is not None
    if _default_session is not None: _default_session.TryClose()
    _default_session = None
    
def try_init_default_session(func):
    @functools.wraps(func)
    def Func(*args):
        GetDefaultSession().TryInit()
        return func(*args)
    return Func

_default_session = None
