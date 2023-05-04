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
#ifndef ONEFLOW_OPENCL_COMMON_CL_CONTEXT_H_
#define ONEFLOW_OPENCL_COMMON_CL_CONTEXT_H_

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "oneflow/opencl/common/cl_kernel_pool.h"
#include "oneflow/opencl/common/CL/opencl.hpp"

namespace oneflow {

typedef struct clContext {
  int device_id;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue default_queue;
  clKernelPool kernel_pool;
} clContext;

class clContextPool {
 public:
  static clContextPool* Get();

  cl_int getOrCreateContext(clContext** context, int device_id);

  cl_int getDevices(cl::Device** devices, int* device_count);

 private:
  clContextPool();

 private:
  mutable std::mutex mutex_;
  cl::Platform platform_;
  std::vector<cl::Device> devices_;

  std::map<int, std::unique_ptr<clContext>> contexts_;
};

}  // namespace oneflow

#endif  // ONEFLOW_OPENCL_COMMON_CL_CONTEXT_H_
