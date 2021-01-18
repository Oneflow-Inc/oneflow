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


def HasSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym):
    global parallel_conf_symbol2shared_opkernel_object
    return parallel_conf_sym in parallel_conf_symbol2shared_opkernel_object


def GetSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym):
    global parallel_conf_symbol2shared_opkernel_object
    return parallel_conf_symbol2shared_opkernel_object[parallel_conf_sym]


def SetSharedOpKernelObject4ParallelConfSymbol(
    parallel_conf_sym, shared_opkernel_object
):
    assert not HasSharedOpKernelObject4ParallelConfSymbol(parallel_conf_sym)
    global parallel_conf_symbol2shared_opkernel_object
    parallel_conf_symbol2shared_opkernel_object[
        parallel_conf_sym
    ] = shared_opkernel_object


parallel_conf_symbol2shared_opkernel_object = {}
