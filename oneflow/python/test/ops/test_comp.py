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
import oneflow
import oneflow.python.ops.utils.compile as compi

py_sigmoid_op_compi = compi.UserOpCompiler("py_sigmoid")
py_sigmoid_op_compi.AddOpDef()
py_sigmoid_op_compi.AddPythonKernel()
py_sigmoid_op_compi.Finish()

user_ops_ld = compi.UserOpsLoader()
user_ops_ld.LoadAll()
