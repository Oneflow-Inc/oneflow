import oneflow.python.framework.oneflow_mode as oneflow_mode
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.runtime as runtime

def run():
    with oneflow_mode.CompileMode():
        job_set = compiler.Compile()
    main = compiler.GetMainFunc()
    with oneflow_mode.RuntimeMode():
        with runtime.MachineRuntimeContext(job_set):
            main()
