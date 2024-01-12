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
import inspect
import time
from functools import wraps

import oneflow as flow

from .log_utils import logger


class cost_cnt:
    def __init__(self, debug=False, message="\t"):
        self.debug = debug
        self.message = message

    def __enter__(self):
        if not self.debug:
            return
        flow._oneflow_internal.eager.Sync()
        before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        before_host_used = flow._oneflow_internal.GetCPUMemoryUsed()
        logger.debug(f"====> {self.message} try to run...")
        logger.debug(f"{self.message} cuda mem before {before_used} MB")
        logger.debug(f"{self.message} host mem before {before_host_used} MB")
        self.before_used = before_used
        self.before_host_used = before_host_used
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.debug:
            return
        flow._oneflow_internal.eager.Sync()
        end_time = time.time()
        after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        after_host_used = flow._oneflow_internal.GetCPUMemoryUsed()
        logger.debug(f"{self.message} run time {end_time - self.start_time} seconds")
        logger.debug(f"{self.message} cuda mem after {after_used} MB")
        logger.debug(f"{self.message} cuda mem diff {after_used - self.before_used} MB")
        logger.debug(f"{self.message} host mem after {after_host_used} MB")
        logger.debug(
            f"{self.message} host mem diff {after_host_used - self.before_host_used} MB"
        )
        logger.debug(f"<==== {self.message} finish run.")

    def __call__(self, func):
        @wraps(func)
        def clocked(*args, **kwargs):
            if not self.debug:
                return func(*args, **kwargs)
            module = inspect.getmodule(func)
            logger.debug(
                f"==> function {module.__name__}.{func.__name__}  try to run..."
            )
            flow._oneflow_internal.eager.Sync()

            before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
            logger.debug(f"{func.__name__} cuda mem before {before_used} MB")

            before_host_used = flow._oneflow_internal.GetCPUMemoryUsed()
            logger.debug(f"{func.__name__} host mem before {before_host_used} MB")

            start_time = time.time()
            out = func(*args, **kwargs)
            flow._oneflow_internal.eager.Sync()
            end_time = time.time()

            logger.debug(f"{func.__name__} run time {end_time - start_time} seconds")

            after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
            logger.debug(f"{func.__name__} cuda mem after {after_used} MB")

            logger.debug(f"{func.__name__} cuda mem diff {after_used - before_used} MB")
            after_host_used = flow._oneflow_internal.GetCPUMemoryUsed()
            logger.debug(f"{func.__name__} host mem after {after_host_used} MB")
            logger.debug(
                f"{func.__name__} host mem diff {after_host_used - before_host_used} MB"
            )

            logger.debug(f"<== function {func.__name__} finish run.")
            logger.debug("")
            return out

        return clocked
