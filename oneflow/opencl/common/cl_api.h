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
#ifndef ONEFLOW_OPENCL_COMMON_CL_API_H_
#define ONEFLOW_OPENCL_COMMON_CL_API_H_

#include <string>
#include "oneflow/core/ep/include/primitive/memcpy.h"  // MemcpyKind

#include "oneflow/opencl/common/CL/opencl.hpp"

namespace oneflow {

using ::oneflow::ep::primitive::MemcpyKind;

cl_int clBuildKernel(const std::string& program_name, const std::string& kernel_name,
                     cl::Kernel* kernel, const std::string& build_options);

cl_int clLaunchKernel(const cl::Kernel& kernel, const cl::NDRange& offset,
                      const cl::NDRange& global, const cl::NDRange& local = cl::NullRange,
                      cl::CommandQueue* queue = 0);

cl_int clGetDeviceCount(int* count);

cl_int clGetDevice(int* device_id);
cl_int clSetDevice(int device_id);

cl_int clMalloc(void** buf, size_t size);
cl_int clFree(void* buf);

cl_int clMallocHost(void** buf, size_t size);
cl_int clFreeHost(void* buf);

cl_int clMemcpy(void* dst, const void* src, size_t size, MemcpyKind kind);
cl_int clMemcpyAsync(void* dst, const void* src, size_t size, MemcpyKind kind,
                     cl::CommandQueue* queue);

cl_int clMemset(void* ptr, int value, size_t size);
cl_int clMemsetAsync(void* ptr, int value, size_t size, cl::CommandQueue* queue);

cl_int clEventCreateWithFlags(cl::Event** event, unsigned int flags);
cl_int clEventDestroy(cl::Event* event);
cl_int clEventRecord(cl::Event* event, cl::CommandQueue* queue);
cl_int clEventQuery(cl::Event* event);
cl_int clEventSynchronize(cl::Event* event);

cl_int clQueueCreate(cl::CommandQueue** queue);
cl_int clQueueDestroy(cl::CommandQueue* queue);
cl_int clQueueSynchronize(cl::CommandQueue* queue);
cl_int clQueueWaitEvent(cl::Event* event, cl::CommandQueue* queue, unsigned int flags = 0);

}  // namespace oneflow

#endif  // ONEFLOW_OPENCL_COMMON_CL_API_H_
