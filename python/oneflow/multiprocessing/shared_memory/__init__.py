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
import oneflow as flow

__all__ = ["SharedMemory"]


class SharedMemory:
    def __init__(self, name=None, create=False, size=0):
        if not size >= 0:
            raise ValueError("'size' must be a non-negative integer")
        if create:
            if size == 0:
                raise ValueError("'size' must be a positive number different from zero")
        self.shm_ = flow._oneflow_internal.multiprocessing.SharedMemory(
            name=name if name is not None else "", create=create, size=size
        )

    def __del__(self):
        try:
            if hasattr(self, "shm_"):
                self.close()
        except OSError:
            pass

    def __reduce__(self):
        return (
            self.__class__,
            (self.name, False, self.size,),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, size={self.size})"

    @property
    def buf(self):
        "A memoryview of contents of the shared memory block."
        return self.shm_.buf

    @property
    def name(self):
        "Unique name that identifies the shared memory block."
        return self.shm_.name

    @property
    def size(self):
        "Size in bytes."
        return self.shm_.size

    def close(self):
        """Closes access to the shared memory from this instance but does
        not destroy the shared memory block."""
        return self.shm_.close()

    def unlink(self):
        """Requests that the underlying shared memory block be destroyed.
        In order to ensure proper cleanup of resources, unlink should be
        called once (and only once) across all processes which have access
        to the shared memory block."""
        return self.shm_.unlink()
