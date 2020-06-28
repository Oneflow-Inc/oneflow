from __future__ import absolute_import


def GetIdForRegisteredCallback(cb):
    assert callable(cb)
    global unique_id2handler
    unique_id2handler[id(cb)] = cb
    return id(cb)


def DeleteRegisteredCallback(cb):
    global unique_id2handler
    assert id(cb) in unique_id2handler
    del unique_id2handler[id(cb)]


unique_id2handler = {}
