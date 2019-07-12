from __future__ import absolute_import

import oneflow.python.framework.oneflow_mode as oneflow_mode
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.runtime as runtime

def compile_only():
    with oneflow_mode.CompileMode():
         return compiler.Compile()

def run():
    job_set = compile_only()
    main = compiler.GetMainFunc()
    with oneflow_mode.RuntimeMode():
        try:
            with runtime.GetMachineRuntimeEnv(job_set):
                main()
        except runtime.ThisIsNotAnError:
            pass
