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
from oneflow.compatible import single_client as flow


class Function:
    def __init__(self, func_name, handle):
        self.func_name = func_name
        self.handle = handle

    def __call__(self, *args, **kwargs):
        return self.handle(*args, **kwargs)


_function_name_aliases = {"add_scalar": "scalar_add"}
