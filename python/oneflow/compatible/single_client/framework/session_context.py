import functools
from oneflow.compatible import single_client as flow
import oneflow._oneflow_internal

class SessionStatus:
    OPEN = 'OPEN'
    RUNNING = 'RUNNING'
    CLOSED = 'CLOSED'

def GetDefaultSession():
    global _sess_id2sess
    default_sess_id = oneflow._oneflow_internal.GetDefaultSessionId()
    assert default_sess_id in _sess_id2sess
    return _sess_id2sess[default_sess_id]

def OpenDefaultSession(sess):
    global _sess_id2sess
    assert sess.id not in _sess_id2sess
    _sess_id2sess[sess.id] = sess

def TryCloseDefaultSession():
    global _sess_id2sess
    default_sess_id = oneflow._oneflow_internal.GetDefaultSessionId()
    assert default_sess_id in _sess_id2sess
    if default_sess_id in _sess_id2sess:
        _sess_id2sess[default_sess_id].TryClose()
    del _sess_id2sess[default_sess_id]

def TryCloseAllSession():
    global _sess_id2sess
    for sess_id in _sess_id2sess.keys():
        _sess_id2sess[sess_id].TryClose()
    _sess_id2sess.clear()

def try_init_default_session(func):

    @functools.wraps(func)
    def Func(*args, **kwargs):
        GetDefaultSession().TryInit()
        return func(*args, **kwargs)
    return Func
_sess_id2sess = {}