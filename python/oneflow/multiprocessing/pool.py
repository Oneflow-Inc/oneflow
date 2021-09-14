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
import multiprocessing.pool
import multiprocessing.util as util

from .queue import SimpleQueue


def clean_worker(*args, **kwargs):
    import gc

    multiprocessing.pool.worker(*args, **kwargs)
    # Regular multiprocessing workers don't fully clean up after themselves,
    # so we have to explicitly trigger garbage collection to make sure that all
    # destructors are called...
    gc.collect()


class Pool(multiprocessing.pool.Pool):
    """Pool implementation which uses our version of SimpleQueue.
    This lets us pass tensors in shared memory across processes instead of
    serializing the underlying data."""

    def _setup_queues(self):
        self._inqueue = SimpleQueue()
        self._outqueue = SimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv

    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(self._processes - len(self._pool)):
            # changed worker -> clean_worker
            args = (
                self._inqueue,
                self._outqueue,
                self._initializer,
                self._initargs,
                self._maxtasksperchild,
            )
            if hasattr(self, "_wrap_exception"):
                args += (self._wrap_exception,)
            w = self.Process(target=clean_worker, args=args)
            self._pool.append(w)
            w.name = w.name.replace("Process", "PoolWorker")
            w.daemon = True
            w.start()
            util.debug("added worker")
