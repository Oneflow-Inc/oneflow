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

import importlib.util
import os
import os.path
import shutil
import subprocess as sp
import sys
import sysconfig
import numpy

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.sysconfig as oneflow_sysconfig
import oneflow
import oneflow_api


def run_cmd(cmd, cwd=None):
    if cwd:
        res = sp.run(cmd, cwd=cwd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    else:
        res = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    out = res.stdout.decode("utf8")
    if res.returncode != 0:
        err_msg = "Run cmd failed: {}, output: {}".format(cmd, out)
        raise Exception(err_msg)
    if len(out) and out[-1] == "\n":
        out = out[:-1]
    return out


def compile(compiler, flags, link, inputs, output):
    if os.path.exists(output):
        return True
    if isinstance(inputs, list):
        cmd = "{} {} {} {} -o {}".format(
            compiler, " ".join(inputs), flags, link, output
        )
    else:
        cmd = "{} {} {} {} -o {}".format(compiler, inputs, flags, link, output)
    print(cmd)
    run_cmd(cmd)
    return True


def get_cflags():
    return " ".join(oneflow_sysconfig.get_compile_flags())


def get_lflags():
    return (
        " ".join(oneflow_sysconfig.get_link_flags())
        + " -Wl,-rpath "
        + oneflow_sysconfig.get_lib()
    )


class PythonKernelRegistry(object):
    """A helper class to store python kernel module
    """

    def __init__(self):
        self.kernels_ = {}

    def Register(self, op_type_name, module):
        self.kernels_[op_type_name] = module


_python_kernel_reg = PythonKernelRegistry()
oneflow_api.RegisterPyKernels(_python_kernel_reg.kernels_)


@oneflow_export("experimental.op_lib")
class OpLib(object):
    def __init__(self, op_type_name, lib_path=""):
        self.op_type_name_ = op_type_name
        self.api = None
        self.so_path_ = ""
        self.objs_ = []
        self.has_api_ = False
        self.has_def_ = False
        self.has_py_kernel_ = False
        self.has_cpu_kernel_ = False
        self.has_gpu_kernel_ = False
        self.got_so_ = False

        lib_path = os.path.normpath(lib_path)
        pwd_path = os.getcwd()
        if lib_path != "." and lib_path != pwd_path:
            lib_folder = os.path.join(lib_path, self.op_type_name_)
            pwd_folder = os.path.join(pwd_path, self.op_type_name_)
            if os.path.exists(pwd_folder):
                shutil.rmtree(pwd_folder)
            shutil.copytree(lib_folder, pwd_folder)

        self.src_prefix_ = os.path.join(
            pwd_path, self.op_type_name_, self.op_type_name_
        )

        out_path = os.path.join(pwd_path, self.op_type_name_, "out")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.out_prefix_ = os.path.join(out_path, self.op_type_name_)

    def py_api(self):
        assert os.path.exists("{}_py_api.py".format(self.src_prefix_))
        spec = importlib.util.spec_from_file_location(
            self.op_type_name_, "{}_py_api.py".format(self.src_prefix_)
        )
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)
        return self

    def cpp_def(self):
        flags = "-std=c++11 -c -fPIC -O2 " + get_cflags()
        compile(
            "g++",
            flags,
            get_lflags(),
            "{}_cpp_def.cpp".format(self.src_prefix_),
            "{}_cpp_def.o".format(self.out_prefix_),
        )
        self.objs_.append("{}_cpp_def.o".format(self.out_prefix_))
        self.has_def_ = True
        return self

    def py_kernel(self):
        assert os.path.exists("{}_py_kernel.py".format(self.src_prefix_))
        spec = importlib.util.spec_from_file_location(
            self.op_type_name_, "{}_py_kernel.py".format(self.src_prefix_)
        )
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
        _python_kernel_reg.Register(self.op_type_name_, kernel)
        oneflow_api.RegisterPyKernelCaller(self.op_type_name_)
        self.has_py_kernel_ = True
        return self

    def cpp_kernel(self):
        flags = "-std=c++11 -c -fPIC -O2 " + get_cflags()
        compile(
            "g++",
            flags,
            "",
            "{}_cpp_kernel.cpp".format(self.src_prefix_),
            "{}_cpp_kernel.o".format(self.out_prefix_),
        )
        self.objs_.append("{}_cpp_kernel.o".format(self.out_prefix_))
        self.has_cpu_kernel_ = True
        return self

    def gpu_kernel(self):
        raise NotImplementedError

    def build_load(self):
        if len(self.objs_) > 0:
            flags = "-std=c++11 -shared -fPIC " + get_cflags()
            compile(
                "g++", flags, get_lflags(), self.objs_, "{}.so".format(self.out_prefix_)
            )
            self.got_so_ = True
            self.so_path_ = self.out_prefix_ + ".so"

        oneflow.config.load_library_now(self.so_path_)
