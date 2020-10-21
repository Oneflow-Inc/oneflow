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


def run_cmd(cmd, cwd=None):
    print(cmd)
    if cwd:
        res = sp.run(cmd, cwd=cwd, shell=True,
                     stdout=sp.PIPE, stderr=sp.STDOUT)
    else:
        res = sp.run(cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT)
    out = res.stdout.decode("utf8")
    if res.returncode != 0:
        err_msg = f"Run cmd failed: {cmd}, output: {out}"
        raise Exception(err_msg)
    if len(out) and out[-1] == "\n":
        out = out[:-1]
    return out


def compile(compiler, flags, link, inputs, output):
    if os.path.exists(output):
        return True
    if isinstance(inputs, list):
        cmd = f"{compiler} {' '.join(inputs)} {flags} {link} -o {output}"
    else:
        cmd = f"{compiler} {inputs} {flags} {link} -o {output}"
    run_cmd(cmd)
    return True


cflags = " ".join(oneflow_sysconfig.get_compile_flags())
lflags = (
    " ".join(oneflow_sysconfig.get_link_flags())
    + " -Wl,-rpath "
    + oneflow_sysconfig.get_lib()
)
cpp2py_path = os.path.join(
    oneflow_sysconfig.get_include(), "oneflow/python/ops/util/cpp2py.hpp"
)


@oneflow_export("util.op_lib")
class OpLib(object):
    def __init__(self, op_type_name, lib_path=""):
        self.op_type_name = op_type_name
        self.objs = []
        self.has_def = False
        self.has_py_kernel = False
        self.has_cpu_kernel = False
        self.got_so = False
        self.api = None
        print(run_cmd("g++ -v"))
        print(run_cmd("nm /root/.local/lib/python3.6/site-packages/oneflow/_oneflow_internal.cpython-36m-x86_64-linux-gnu.so | grep CheckAndGetOpRegistry"))

        lib_path = os.path.normpath(lib_path)
        pwd_path = os.getcwd()
        if lib_path != "" and lib_path != pwd_path:
            lib_folder = os.path.join(lib_path, self.op_type_name)
            pwd_folder = os.path.join(pwd_path, self.op_type_name)
            if os.path.exists(pwd_folder):
                shutil.rmtree(pwd_folder)
            shutil.copytree(lib_folder, pwd_folder)

        self.src_prefix = os.path.join(
            pwd_path, self.op_type_name, self.op_type_name)

        out_path = os.path.join(pwd_path, self.op_type_name, "out")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.out_prefix = os.path.join(out_path, self.op_type_name)
        self.so_path = ""

    def AddOpDef(self):
        flags = "-std=c++11 -c -fPIC -O2 " + cflags
        compile(
            "g++",
            flags,
            lflags,
            f"{self.src_prefix}_op.cpp",
            f"{self.out_prefix}_op.o",
        )
        self.objs.append(f"{self.out_prefix}_op.o")
        self.has_def = True
        return True

    def AddPythonAPI(self):
        spec = importlib.util.spec_from_file_location(
            self.op_type_name, f"{self.src_prefix}_py_api.py"
        )
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)

    def AddPythonKernel(self):
        assert os.path.exists(f"{self.src_prefix}_py_kernel.py")
        self.has_py_kernel = True

    def AddCPUKernel(self):
        flags = "-std=c++11 -c -fPIC -O2 " + cflags
        compile(
            "g++",
            flags,
            "",
            f"{self.src_prefix}_cpu_kernel.cpp",
            f"{self.out_prefix}_cpu_kernel.o",
        )
        self.objs.append(f"{self.out_prefix}_cpu_kernel.o")
        self.has_cpu_kernel = True
        return True

    def AddGPUKernel(self):
        raise NotImplementedError

    def Build(self):
        if len(self.objs) > 0:
            flags = "-std=c++11 -shared -fPIC " + cflags
            compile("g++", flags, lflags, self.objs, f"{self.out_prefix}.so")
            self.got_so = True
            self.so_path = self.out_prefix + ".so"

        return self.so_path


@oneflow_export("util.op_lib_loader")
class OpLibLoader(object):
    def __init__(self):
        self.py_names = []
        self.so_names = []
        self.lib_names = []
        self.py_kernel_lib = ""
        self.linked = False

    def AddLib(self, op_lib_builder):
        if op_lib_builder.got_so:
            self.so_names.append(op_lib_builder.so_path)
        if op_lib_builder.has_py_kernel:
            self.py_names.append(op_lib_builder.op_type_name)

    def Link(self):
        for so in self.so_names:
            self.lib_names.append(so)

        if len(self.py_names) > 0:

            def get_one_reg_src(op_type_name):
                one_reg_src = f"""
                #define REGISTER_{op_type_name}_KERNEL(cpp_type, dtype) REGISTER_USER_KERNEL("{op_type_name}").SetCreateFn<PyKernel<cpp_type>>().SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") & (user_op::HobDataType() == dtype));
            
                OF_PP_FOR_EACH_TUPLE(REGISTER_{op_type_name}_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);
            
                #define REGISTER_{op_type_name}_GRAD_KERNEL(cpp_type, dtype) REGISTER_USER_KERNEL("{op_type_name}_grad").SetCreateFn<PyGradKernel<cpp_type>>().SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") & (user_op::HobDataType() == dtype));
            
                OF_PP_FOR_EACH_TUPLE(REGISTER_{op_type_name}_GRAD_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);
                """
                return one_reg_src

            def get_reg_src():
                reg_src = []
                for op in self.py_names:
                    reg_src.append(get_one_reg_src(op))
                full_reg_src = f"""
                namespace oneflow {{
                    {"".join(reg_src)}
                }}  // namespace oneflow
                """
                return full_reg_src

            self.out_path = os.path.join(os.getcwd(), "op_lib_loader_out")
            if os.path.exists(self.out_path):
                shutil.rmtree(self.out_path)
            os.makedirs(self.out_path)
            gen_src_path = os.path.join(self.out_path, "cpp2py_gen.cpp")
            gen_obj_path = os.path.join(self.out_path, "cpp2py.o")
            gen_so_path = os.path.join(self.out_path, "cpp2py.so")
            shutil.copy2(cpp2py_path, gen_src_path)
            with open(gen_src_path, "a") as f:
                f.write(get_reg_src())
            # compile obj
            py_cflags = "-std=c++11 -c -fPIC -O2 " + cflags
            py_cflags += " -I" + sysconfig.get_paths()["include"]
            py_cflags += " -I" + numpy.get_include()
            compile("g++", py_cflags, "", gen_src_path, gen_obj_path)
            # compile so
            py_cflags = "-std=c++11 -shared -fPIC " + cflags
            py_cflags += " -I" + sysconfig.get_paths()["include"]
            py_cflags += " -I" + numpy.get_include()
            py_lflags = lflags
            py_lflags += " -L" + sysconfig.get_paths()["stdlib"]
            compile("g++", py_cflags, py_lflags, gen_obj_path, gen_so_path)
            self.py_kernel_lib = gen_so_path
            self.lib_names.append(gen_so_path)
        self.linked = True

    def Load(self):
        assert self.linked
        for lib in self.lib_names:
            oneflow.config.load_library(lib)

    def LibList(self):
        assert self.linked
        return self.lib_names

    def PythonKernelLib(self):
        assert self.linked
        return self.py_kernel_lib
