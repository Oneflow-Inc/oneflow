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
from oneflow.compatible.single_client.framework import hob as hob
from oneflow.compatible.single_client.support import enable_if as enable_if


def api_enable_typing_check(val: bool = True) -> None:
    """ enable typing check for global_function """
    return enable_if.unique([enable_typing_check])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.any_global_function_defined)
def enable_typing_check(val):
    global typing_check_enabled
    typing_check_enabled = val


typing_check_enabled = False
