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
#ifndef ONEFLOW_EXTENSION_PYTHON_PY_COMPUTE_H_
#define ONEFLOW_EXTENSION_PYTHON_PY_COMPUTE_H_
#include <Python.h>
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace pyext {
void PyRegisterKernels(PyObject* py_kernels);
void PyCompute(user_op::KernelComputeContext* ctx, const std::string& py_func_name);
}  // namespace pyext
}  // namespace oneflow

#endif  // ONEFLOW_EXTENSION_PYTHON_PY_COMPUTE_H_
