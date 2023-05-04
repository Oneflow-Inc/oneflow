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
#ifndef ONEFLOW_OPENCL_COMMON_CL_UTIL_H_
#define ONEFLOW_OPENCL_COMMON_CL_UTIL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ep/include/primitive/memcpy.h"  // MemcpyKind

#include "oneflow/opencl/common/cl_api.h"

#define CL_CHECK_OR_RETURN(expr)           \
  {                                        \
    cl_int ret = (expr);                   \
    if (ret != CL_SUCCESS) { return ret; } \
  }

#define OF_CL_CHECK(condition)                                                     \
  for (cl_int _cnrt_check_status = (condition); _cnrt_check_status != CL_SUCCESS;) \
  THROW(RuntimeError) << "OpenCL check failed: " #condition " : "                  \
                      << " (" << _cnrt_check_status << ") "

#endif  // ONEFLOW_OPENCL_COMMON_CL_UTIL_H_
