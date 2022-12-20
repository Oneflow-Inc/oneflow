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
from typing import Any, Optional


class autocast(flow.amp.autocast_mode.autocast):
    r"""
    See :class:`oneflow.autocast`.
    ``oneflow.cuda.amp.autocast(args...)`` is equivalent to ``oneflow.autocast("cuda", args...)``
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: Optional[flow.dtype] = None,
        cache_enabled: Optional[bool] = None,
    ):
        super().__init__(
            "cuda", enabled=enabled, dtype=dtype, cache_enabled=cache_enabled
        )

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        return super().__call__(func)
