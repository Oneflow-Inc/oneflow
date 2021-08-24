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
import traceback
import oneflow._oneflow_internal


class BlockingInfoContext:
    def __init__(self, save_stack=True):
        self.save_stack_ = save_stack
        stack_info = "".join(traceback.format_stack(limit=5))
        oneflow._oneflow_internal.blocking.register_stack_info_callback(
            lambda: stack_info
        )

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        oneflow._oneflow_internal.blocking.clear_stack_info_callback()
