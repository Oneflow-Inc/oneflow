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
