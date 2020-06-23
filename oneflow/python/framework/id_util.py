from __future__ import absolute_import


def UniqueStr(prefix):
    return "%s%d" % (prefix, UniqueId())


def UniqueId():
    global _unique_id
    ret = _unique_id
    _unique_id += 1
    return ret


_unique_id = 0
