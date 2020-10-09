import oneflow
import oneflow.python.ops.utils.compile as compi

py_sigmoid_op_compi = compi.UserOpCompiler("py_sigmoid")
py_sigmoid_op_compi.AddOpDef()
py_sigmoid_op_compi.AddPythonKernel()
py_sigmoid_op_compi.Finish()

user_ops_ld = compi.UserOpsLoader()
user_ops_ld.LoadAll()
