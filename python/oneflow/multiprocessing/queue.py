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
import io
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
import pickle


class ConnectionWrapper(object):
    """Proxy class for _multiprocessing.Connection which uses ForkingPickler to
    serialize objects"""

    def __init__(self, conn):
        self.conn = conn

    def send(self, obj):
        buf = io.BytesIO()
        ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(obj)
        self.send_bytes(buf.getvalue())

    def recv(self):
        buf = self.recv_bytes()
        return pickle.loads(buf)

    def __getattr__(self, name):
        if "conn" in self.__dict__:
            return getattr(self.conn, name)
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, "conn")
        )


class Queue(multiprocessing.queues.Queue):
    def __init__(self, *args, **kwargs):
        super(Queue, self).__init__(*args, **kwargs)
        self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
        self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        self._send = self._writer.send
        self._recv = self._reader.recv


class SimpleQueue(multiprocessing.queues.SimpleQueue):
    def _make_methods(self):
        if not isinstance(self._reader, ConnectionWrapper):
            self._reader: ConnectionWrapper = ConnectionWrapper(self._reader)
            self._writer: ConnectionWrapper = ConnectionWrapper(self._writer)
        super(SimpleQueue, self)._make_methods()  # type: ignore[misc]
