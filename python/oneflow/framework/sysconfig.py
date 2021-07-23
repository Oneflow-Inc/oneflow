import imp
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


def get_link_flags() -> List[str]:
    flags = []
    flags.append("-L{}".format(get_lib()))
    (file, oneflow_internal_lib_path, _) = imp.find_module(
        "_oneflow_internal", [get_lib()]
    )
    if file:
        file.close()
    flags.append("-l:{}".format(os.path.basename(oneflow_internal_lib_path)))
    return flags


def with_cuda() -> bool:
    return oneflow._oneflow_internal.flags.with_cuda()


def with_xla() -> bool:
    return oneflow._oneflow_internal.flags.with_xla()


def has_rpc_backend_grpc() -> bool:
    return oneflow._oneflow_internal.flags.has_rpc_backend_grpc()


def has_rpc_backend_local() -> bool:
    return oneflow._oneflow_internal.flags.has_rpc_backend_local()


def cmake_build_type() -> str:
    return oneflow._oneflow_internal.flags.cmake_build_type()
