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
import importlib.machinery
import importlib.util
import os
from typing import List

import oneflow
import oneflow._oneflow_internal


def get_include() -> str:
    return os.path.join(os.path.dirname(oneflow.__file__), "include")


def get_lib() -> str:
    return os.path.dirname(oneflow.__file__)


def get_compile_flags() -> List[str]:
    flags = []
    flags.append("-I{}".format(get_include()))
    flags.append("-DHALF_ENABLE_CPP11_USER_LITERALS=0")
    if oneflow._oneflow_internal.flags.with_cuda():
        flags.append("-DWITH_CUDA")
    if oneflow._oneflow_internal.flags.use_cxx11_abi():
        flags.append("-D_GLIBCXX_USE_CXX11_ABI=1")
    else:
        flags.append("-D_GLIBCXX_USE_CXX11_ABI=0")
    return flags


def get_liboneflow_link_flags() -> List[str]:
    oneflow_python_module_path = get_lib()
    # path in a pip release
    oneflow_python_libs_path = f"{oneflow_python_module_path}.libs"
    # path in a cmake build dir
    if not os.path.exists(oneflow_python_libs_path):
        from oneflow.version import __cmake_project_binary_dir__

        oneflow_python_libs_path = __cmake_project_binary_dir__
    return [
        f"-L{oneflow_python_libs_path}",
        f"-l:oneflow",
        f"-l:of_protoobj",
    ]


def get_link_flags() -> List[str]:
    flags = []
    flags.append("-L{}".format(get_lib()))
    # imp.find_module was removed in Python 3.12; locate the extension by its known
    # suffixes under get_lib() (e.g. _oneflow_internal.cpython-3xx-<plat>.so).
    lib_dir = get_lib()
    oneflow_internal_lib_path = next(
        os.path.join(lib_dir, "_oneflow_internal" + suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if os.path.exists(os.path.join(lib_dir, "_oneflow_internal" + suffix))
    )
    flags.append("-l:{}".format(os.path.basename(oneflow_internal_lib_path)))
    return flags


def with_cuda() -> bool:
    return oneflow._oneflow_internal.flags.with_cuda()


def get_cuda_version() -> int:
    return oneflow._oneflow_internal.flags.cuda_version()


def has_rpc_backend_grpc() -> bool:
    return oneflow._oneflow_internal.flags.has_rpc_backend_grpc()


def has_rpc_backend_local() -> bool:
    return oneflow._oneflow_internal.flags.has_rpc_backend_local()


def cmake_build_type() -> str:
    return oneflow._oneflow_internal.flags.cmake_build_type()


def with_rdma() -> bool:
    return oneflow._oneflow_internal.flags.with_rdma()
