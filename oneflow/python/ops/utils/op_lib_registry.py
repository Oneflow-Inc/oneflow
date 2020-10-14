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
import importlib.util
import os
import os.path
import shutil
import subprocess as sp
import sys
import sysconfig
import numpy

import oneflow


def run_cmd(cmd, cwd=None):
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
    print(cmd)
    run_cmd(cmd)
    return True


cflags = " ".join(oneflow.sysconfig.get_compile_flags())
lflags = (
    " ".join(oneflow.sysconfig.get_link_flags())
    + " -Wl,-rpath "
    + oneflow.sysconfig.get_lib()
)
cpp2py_path = os.path.join(oneflow.sysconfig.get_lib(),
                           "python/ops/utils/cpp2py.hpp")


class OpLib(object):
    def __init__(self, op_type_name):
        self.op_type_name = op_type_name
        self.objs = []
        self.has_def = False
        self.has_py_kernel = False
        self.has_cpu_kernel = False
        self.got_so = False
        self.api = None

    def AddOpDef(self):
        flags = "-std=c++11 -c -fPIC -O2 " + cflags
        compile(
            "g++",
            flags,
            lflags,
            f"{self.op_type_name}_op.cpp",
            f"{self.op_type_name}_op.o",
        )
        self.objs.append(f"{self.op_type_name}_op.o")
        self.has_def = True
        return True

    def AddPythonAPI(self):
        spec = importlib.util.spec_from_file_location(
            self.op_type_name, f"./{self.op_type_name}_py_api.py"
        )
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)
        print(spec)
        print(self.api)

    def AddPythonKernel(self):
        assert os.path.exists(f"{self.op_type_name}_py_kernel.py")
        self.has_py_kernel = True

    def AddCPUKernel(self):
        flags = "-std=c++11 -c -fPIC -O2 " + cflags
        compile(
            "g++",
            flags,
            "",
            f"{self.op_type_name}_cpu_kernel.cpp",
            f"{self.op_type_name}_cpu_kernel.o",
        )
        self.objs.append(f"{self.op_type_name}_cpu_kernel.o")
        self.has_cpu_kernel = True
        return True

    def AddGPUKernel(self):
        raise NotImplementedError

    def Build(self):
        if len(self.objs) > 0:
            flags = "-std=c++11 -shared -fPIC " + cflags
            compile("g++", flags, lflags, self.objs, f"{self.op_type_name}.so")
            self.got_so = True
            return os.getcwd() + "/" + self.op_type_name + ".so"
        else:
            return ""


class OpLibLoader(object):
    def __init__(self):
        self.py_names = []
        self.so_names = []
        self.lib_names = []
        self.linked = False

    def AddLib(self, op_lib_builder):
        if op_lib_builder.got_so:
            self.so_names.append(op_lib_builder.op_type_name)
        if op_lib_builder.has_py_kernel:
            self.py_names.append(op_lib_builder.op_type_name)

    def Link(self):
        for op in self.so_names:
            self.lib_names.append(os.getcwd() + "/" + op + ".so")

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

            shutil.copy2(cpp2py_path, "./cpp2py_tmp.cpp")
            with open("./cpp2py_tmp.cpp", "a") as f:
                f.write(get_reg_src())
            # compile obj
            py_cflags = "-std=c++11 -c -fPIC -O2 " + cflags
            py_cflags += " -I" + sysconfig.get_paths()["include"]
            py_cflags += " -I" + numpy.get_include()
            compile("g++", py_cflags, "", "cpp2py_tmp.cpp", "cpp2py.o")
            # compile so
            py_cflags = "-std=c++11 -shared -fPIC " + cflags
            py_cflags += " -I" + sysconfig.get_paths()["include"]
            py_cflags += " -I" + numpy.get_include()
            py_lflags = lflags
            py_lflags += " -L" + sysconfig.get_paths()["stdlib"]
            compile("g++", py_cflags, py_lflags, "cpp2py.o", "cpp2py.so")
            self.lib_names.append(os.getcwd() + "/cpp2py.so")
        self.linked = True

    def Load(self):
        assert self.linked
        for lib in self.lib_names:
            oneflow.config.load_library(lib)

    def LibList(self):
        assert self.linked
        return self.lib_names
