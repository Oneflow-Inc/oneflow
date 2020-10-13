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
    out = res.stdout.decode('utf8')
    if res.returncode != 0:
        err_msg = f"Run cmd failed: {cmd}, output: {out}"
        raise Exception(err_msg)
    if len(out) and out[-1] == '\n':
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


cflags = ' '.join(oneflow.sysconfig.get_compile_flags())
lflags = ' '.join(oneflow.sysconfig.get_link_flags()) + \
    ' -Wl,-rpath ' + oneflow.sysconfig.get_lib()
cpp2py_path = os.path.join(
    oneflow.sysconfig.get_lib(), "python/ops/utils/cpp2py.hpp")
py_op_types = []
op_so_names = []


class UserOpCompiler(object):
    def __init__(self, op_type_name):
        self.op_type_name = op_type_name
        self.objs = []
        self.has_def = False
        self.has_py_kernel = False
        self.has_cpu_kernel = False

    def AddOpDef(self):
        flags = "-std=c++11 -c -fPIC -O2 " + cflags
        compile("g++", flags, lflags,
                f"{self.op_type_name}_op.cpp", f"{self.op_type_name}_op.o")
        self.objs.append(f"{self.op_type_name}_op.o")
        self.has_def = True
        return True

    def AddPythonKernel(self):
        assert os.path.exists(f"{self.op_type_name}_py_kernel.py")
        self.has_py_kernel = True
        py_op_types.append(self.op_type_name)

    def AddCPUKernel(self):
        flags = "-std=c++11 -c -fPIC -O2 " + cflags
        compile("g++", flags, "",
                f"{self.op_type_name}_cpu_kernel.cpp", f"{self.op_type_name}_cpu_kernel.o")
        self.objs.append(f"{self.op_type_name}_cpu_kernel.o")
        self.has_cpu_kernel = True
        return True

    def AddGPUKernel(self):
        raise NotImplementedError

    def Finish(self):
        if len(self.objs) > 0:
            flags = "-std=c++11 -shared -fPIC " + cflags
            compile("g++", flags, lflags, self.objs, f"{self.op_type_name}.so")
            op_so_names.append(self.op_type_name)
        return True


class UserOpsLoader(object):
    def __init__(self):
        self.so_names = []

    def LoadAll(self):
        for op in op_so_names:
            self.so_names.append(os.getcwd() + "/" + op + ".so")
            oneflow.config.load_library(os.getcwd() + "/" + op + ".so")
        if len(py_op_types) > 0:
            # need to add py_kernel.cpp
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
                for op in py_op_types:
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
            py_cflags += " -I" + sysconfig.get_paths()['include']
            py_cflags += " -I" + numpy.get_include()
            compile("g++", py_cflags, "", "cpp2py_tmp.cpp", "cpp2py.o")
            # compile so
            py_cflags = "-std=c++11 -shared -fPIC " + cflags
            py_cflags += " -I" + sysconfig.get_paths()['include']
            py_cflags += " -I" + numpy.get_include()
            py_lflags = lflags
            py_lflags += " -L" + sysconfig.get_paths()['stdlib']
            compile("g++", py_cflags, py_lflags, "cpp2py.o", "cpp2py.so")
            self.so_names.append(os.getcwd() + "/cpp2py.so")
            oneflow.config.load_library(os.getcwd() + "/cpp2py.so")
