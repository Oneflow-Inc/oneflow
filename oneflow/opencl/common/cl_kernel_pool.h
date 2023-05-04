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
#ifndef ONEFLOW_OPENCL_COMMON_CL_KERNEL_POOL_H_
#define ONEFLOW_OPENCL_COMMON_CL_KERNEL_POOL_H_

#include <map>
#include <mutex>
#include <string>

#include "oneflow/opencl/common/CL/opencl.hpp"

namespace oneflow {

typedef struct clContext clContext;

class clKernelPool {
 public:
  clKernelPool() = default;

  cl_int buildKernel(clContext* context, const std::string& program_name,
                     const std::string& kernel_name, cl::Kernel* kernel,
                     const std::string& build_options = "");

 private:
  mutable std::mutex mutex_;
  std::map<std::tuple<std::string, std::string>, cl::Program> programs_;
  std::map<std::tuple<std::string, std::string, std::string>, cl::Kernel> kernels_;
};

}  // namespace oneflow

#endif  // ONEFLOW_OPENCL_COMMON_CL_KERNEL_POOL_H_
