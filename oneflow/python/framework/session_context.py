"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import functools
import oneflow
import oneflow_api


class SessionStatus:
    OPEN = "OPEN"

    RUNNING = "RUNNING"

    CLOSED = "CLOSED"


def GetDefaultSession():
    global _sess_id2sess
    default_sess_id = oneflow_api.GetDefaultSessionId()
    assert default_sess_id in _sess_id2sess
    return _sess_id2sess[default_sess_id]


def OpenDefaultSession(sess):
    global _sess_id2sess
    assert sess.id not in _sess_id2sess
    _sess_id2sess[sess.id] = sess


def TryCloseDefaultSession():
    global _sess_id2sess
    default_sess_id = oneflow_api.GetDefaultSessionId()
    assert default_sess_id in _sess_id2sess
    if default_sess_id in _sess_id2sess:
        _sess_id2sess[default_sess_id].TryClose()
    del _sess_id2sess[default_sess_id]


def try_init_default_session(func):
    @functools.wraps(func)
    def Func(*args, **kwargs):
        GetDefaultSession().TryInit()
        return func(*args, **kwargs)

    return Func


_sess_id2sess = {}
