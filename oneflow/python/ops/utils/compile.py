import os.path
import subprocess as sp
import sys
import shutil

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
    cmd = f"{compiler} {' '.join(inputs)} {flags} {link} -o {output}"
    run_cmd(cmd)
    return True


cflags = ' '.join(oneflow.sysconfig.get_compile_flags())
lflags = ' '.join(oneflow.sysconfig.get_link_flags())
py_kernel_path = os.path.join(
    oneflow.sysconfig.get_python_cpp_api(), "py_kernel.cpp")
py_op_types = []
op_so_names = []


class UserOpCompiler(object):
    def __init__(self, op_type_name):
        self.op_type_name = op_type_name
        self.objs = []
        self.has_def = False
        self.has_py_kernel = False
        self.has_cpu_kernel = False

    def AddDef(self):
        flags = "-std=c++11 -c -fPIC -O2" + cflags
        compile("g++", flags, "",
                "{self.op_type_name}_op.cpp", "{self.op_type_name}_op.o")
        self.objs.append("{self.op_type_name}_op.o")
        self.has_def = True
        return True

    def AddPythonKernel(self):
        assert os.path.exists("{self.op_type_name}_py_kernel.py")
        self.has_py_kernel = True
        py_op_types.append(self.op_type_name)

    def AddCPUKernel(self):
        flags = "-std=c++11 -c -fPIC -O2" + cflags
        compile("g++", flags, "",
                "{self.op_type_name}_cpu_kernel.cpp", "{self.op_type_name}_cpu_kernel.o")
        self.objs.append("{self.op_type_name}_cpu_kernel.o")
        self.has_cpu_kernel = True
        return True

    def AddGPUKernel(self):
        raise NotImplementedError

    def Finish(self):
        flags = "-std=c++11 -shared -fPIC " + cflags
        compile("g++", flags, lflags, self.objs, "{self.op_type_name}.so")
        op_so_names.append(self.op_type_name)
        return True


class UserOpsLoader(object):
    def __init__(self):
        pass

    def LoadAll(self):
        for op in op_so_names:
            oneflow.config.load_library(op + ".so")
        if len(py_op_types) > 0:
            # need to add py_kernel.cpp
            def get_one_reg_src(op_type_name):
                one_reg_src = f"""
                #define REGISTER_{op_type_name}_KERNEL(cpp_type, dtype)                                     \
                  REGISTER_USER_KERNEL("{op_type_name}").SetCreateFn<PyKernel<cpp_type>>().SetIsMatchedHob( \
                      (user_op::HobDeviceTag() == "cpu"));
            
                OF_PP_FOR_EACH_TUPLE(REGISTER_{op_type_name}_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);
            
                #define REGISTER_{op_type_name}_GRAD_KERNEL(cpp_type, dtype)                                     \
                  REGISTER_USER_KERNEL("{op_type_name}_grad").SetCreateFn<PyGradKernel<cpp_type>>().SetIsMatchedHob( \
                      (user_op::HobDeviceTag() == "cpu"));
            
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

            shutil.copy2(py_kernel_path, "./py_kernel_tmp.cpp")
            with open("./py_kernel_tmp.cpp", "a") as f:
                f.write(get_reg_src)
            flags = "-std=c++11 -c -fPIC -O2" + cflags
            compile("g++", flags, "", "py_kernel_tmp.cpp", "py_kernel.o")
            flags = "-std=c++11 -shared -fPIC " + cflags
            compile("g++", flags, lflags, "py_kernel.o", "py_kernel.so")
            oneflow.config.load_library("py_kernel.so")
