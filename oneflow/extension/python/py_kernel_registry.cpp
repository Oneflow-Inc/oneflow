/*
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
*/
#include "oneflow/extension/python/py_kernel_registry.h"
#include "oneflow/extension/python/py_compute.h"
#include "oneflow/extension/python/py_kernel_caller.h"

namespace oneflow {
namespace pyext {

void RegisterPyKernelCaller(const std::string& op_module_name) {
  // register python op kernel
  auto reg = user_op::UserOpRegistryMgr::Get()
                 .CheckAndGetOpKernelRegistry(op_module_name + "_forward")
                 .SetCreateFn<PyForwardKernel>()
                 .SetIsMatchedHob(
                     ((user_op::HobDeviceTag() == "cpu") & (user_op::HobDeviceSubTag() == "py")));
  user_op::UserOpRegistryMgr::Get().Register(reg.Finish().GetResult());
  // register python grad op kernel
  auto grad_reg = user_op::UserOpRegistryMgr::Get()
                      .CheckAndGetOpKernelRegistry(op_module_name + "_backward")
                      .SetCreateFn<PyBackwardKernel>()
                      .SetIsMatchedHob(((user_op::HobDeviceTag() == "cpu")
                                        & (user_op::HobDeviceSubTag() == "py")));
  user_op::UserOpRegistryMgr::Get().Register(grad_reg.Finish().GetResult());
}

void RegisterPyKernels(PyObject* py_kernels) { PyRegisterKernels(py_kernels); }

}  // namespace pyext

}  // namespace oneflow
