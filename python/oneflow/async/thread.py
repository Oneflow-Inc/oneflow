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
import oneflow._oneflow_internal

class thread:
    def __init__(self, thread_global_id=1, exclude_ccl=False):
        self.stream_set_ = oneflow._oneflow_internal.StreamSet(thread_global_id)
        self.exclude_ccl_ = exclude_ccl

    def __enter__(self):
        self.guard_ = oneflow._oneflow_internal.StreamGuard(self.stream_set_, self.exclude_ccl_)

    def __exit__(self, type, value, traceback):
        del self.guard_
