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

from oneflow.python.oneflow_export import oneflow_export
from oneflow_api.autograd import NoGradGuard


@oneflow_export("no_grad")
class no_grad:
    def __enter__(self):
        self._no_grad_guard = NoGradGuard()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
