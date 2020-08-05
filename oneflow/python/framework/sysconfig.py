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
from __future__ import absolute_import

import os

import oneflow
from oneflow.python.oneflow_export import oneflow_export
from .sysconfig_gen import generated_compile_flags
from typing import List


@oneflow_export("sysconfig.get_include")
def get_include() -> str:
    return os.path.join(os.path.dirname(oneflow.__file__), "include")


@oneflow_export("sysconfig.get_lib")
def get_lib() -> str:
    return os.path.join(os.path.dirname(oneflow.__file__))


@oneflow_export("sysconfig.get_compile_flags")
def get_compile_flags() -> List[str]:
    flags = []
    flags.append("-I%s" % get_include())
    flags.append("-DHALF_ENABLE_CPP11_USER_LITERALS=0")
    flags.extend(generated_compile_flags)
    return flags


@oneflow_export("sysconfig.get_link_flags")
def get_link_flags() -> List[str]:
    flags = []
    flags.append("-L%s" % get_lib())
    flags.append("-l:_oneflow_internal.so")
    return flags
