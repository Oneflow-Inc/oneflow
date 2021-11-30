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
import os
import traceback


def GetFrameLocationStr(depth=-1):
    assert depth < 0
    frame = traceback.extract_stack()[depth - 1]
    return "%s:%d" % (frame[0], frame[1])


def GetStackInfoExcludeOneflowPythonFile():
    import oneflow

    dirname = os.path.dirname(oneflow.__file__)
    stack_info = traceback.extract_stack()
    filtered_stack_info = filter(
        lambda x: x[0].startswith(dirname) == False, stack_info
    )
    return list(filtered_stack_info)
