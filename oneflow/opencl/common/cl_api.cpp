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
#include "oneflow/opencl/common/cl_api.h"

#include <map>
#include <memory>
#include <mutex>
#include "oneflow/opencl/common/cl_context.h"
#include "oneflow/opencl/common/cl_util.h"

namespace oneflow {
namespace {

cl_int clGetContext(clContext** context, int device_id) {
  return clContextPool::Get()->getOrCreateContext(context, device_id);
}

cl_int clGetDevices(cl::Device** devices, int* device_count) {
  return clContextPool::Get()->getDevices(devices, device_count);
}

static thread_local clContext* active_cl_context = nullptr;
static thread_local std::once_flag active_cl_context_inited_flag;

cl_int clGetActiveContext(clContext** context) {
  cl_int ret = CL_SUCCESS;
  std::call_once(active_cl_context_inited_flag,
                 [&]() { ret = clGetContext(&active_cl_context, 0); });
  *context = active_cl_context;
  return ret;
}

cl_int clSetActiveContext(clContext* context) {
  std::call_once(active_cl_context_inited_flag, [&]() { active_cl_context = context; });
  return CL_SUCCESS;
}

std::map<uint64_t, cl::Buffer*>* clGetPinnedMemPool() {
  static std::map<uint64_t, cl::Buffer*> cl_buffer_pool;
  return &cl_buffer_pool;
}

cl_int clPinnedMemRecord(void* host_ptr, cl::Buffer* buffer) {
  if (!clGetPinnedMemPool()->emplace(reinterpret_cast<uint64_t>(host_ptr), buffer).second) {
    return CL_INVALID_HOST_PTR;
  }
  return CL_SUCCESS;
}

cl_int clPinnedMemQuery(void* host_ptr, cl::Buffer** buffer) {
  const auto& it = clGetPinnedMemPool()->find(reinterpret_cast<uint64_t>(host_ptr));
  if (it == clGetPinnedMemPool()->end()) {
    *buffer = nullptr;
    return CL_INVALID_HOST_PTR;
  }
  *buffer = it->second;
  return CL_SUCCESS;
}

cl_int clPinnedMemRelease(void* host_ptr) {
  const auto& it = clGetPinnedMemPool()->find(reinterpret_cast<uint64_t>(host_ptr));
  if (it != clGetPinnedMemPool()->end()) {
    clContext* context = nullptr;
    CL_CHECK_OR_RETURN(clGetActiveContext(&context));
    CL_CHECK_OR_RETURN(context->default_queue.enqueueUnmapMemObject(
        *static_cast<cl::Memory*>(it->second), host_ptr, 0, 0));
    delete it->second;
    clGetPinnedMemPool()->erase(reinterpret_cast<uint64_t>(host_ptr));
  }
  return CL_SUCCESS;
}

bool clIsPinnedMem(void* ptr) {
  const auto& it = clGetPinnedMemPool()->find(reinterpret_cast<uint64_t>(ptr));
  return it != clGetPinnedMemPool()->end();
}

}  // namespace

cl_int clBuildKernel(const std::string& program_name, const std::string& kernel_name,
                     cl::Kernel* kernel, const std::string& build_options) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetActiveContext(&context));
  return context->kernel_pool.buildKernel(context, program_name, kernel_name, kernel,
                                          build_options);
}

cl_int clLaunchKernel(const cl::Kernel& kernel, const cl::NDRange& offset,
                      const cl::NDRange& global, const cl::NDRange& local,
                      cl::CommandQueue* queue) {
  if (!queue) {
    clContext* context = nullptr;
    CL_CHECK_OR_RETURN(clGetActiveContext(&context));
    queue = &(context->default_queue);
  }
  return queue->enqueueNDRangeKernel(kernel, offset, global, local);
}

cl_int clGetDeviceCount(int* count) { return clGetDevices(nullptr, count); }

cl_int clGetDevice(int* device_id) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetActiveContext(&context));
  *device_id = context->device_id;
  return CL_SUCCESS;
}

cl_int clSetDevice(int device_id) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetContext(&context, device_id));
  CL_CHECK_OR_RETURN(clSetActiveContext(context));
  return CL_SUCCESS;
}

cl_int clMalloc(void** buf, size_t size) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetActiveContext(&context));
  cl_int ret = CL_SUCCESS;
  *buf = static_cast<void*>(new cl::Buffer(context->context, CL_MEM_READ_WRITE, size, 0, &ret));
  if (ret != CL_SUCCESS) { *buf = nullptr; }
  return ret;
}

cl_int clFree(void* buf) {
  if (buf) { delete reinterpret_cast<cl::Buffer*>(buf); }
  return CL_SUCCESS;
}

cl_int clMallocHost(void** buf, size_t size) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetActiveContext(&context));
  *buf = nullptr;
  cl_int ret = CL_SUCCESS;
  std::unique_ptr<cl::Buffer> cl_buffer(
      new cl::Buffer(context->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, 0, &ret));
  if (ret != CL_SUCCESS) { return ret; }
  void* host_ptr = context->default_queue.enqueueMapBuffer(
      *cl_buffer, CL_BLOCKING, CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, 0, &ret);
  if (ret != CL_SUCCESS) { return ret; }
  *buf = host_ptr;
  ret = clPinnedMemRecord(host_ptr, cl_buffer.release());
  return ret;
}

cl_int clFreeHost(void* buf) { return clPinnedMemRelease(buf); }

cl_int clMemcpy(void* dst, const void* src, size_t size, MemcpyKind kind) {
  if (size <= 0) { return CL_SUCCESS; }
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetActiveContext(&context));
  CL_CHECK_OR_RETURN(clMemcpyAsync(dst, src, size, kind, &(context->default_queue)));
  // blocking
  context->default_queue.finish();
  return CL_SUCCESS;
}

cl_int clMemcpyAsync(void* dst, const void* src, size_t size, MemcpyKind kind,
                     cl::CommandQueue* queue) {
  if (kind == MemcpyKind::kDtoD) {
    CL_CHECK_OR_RETURN(queue->enqueueCopyBuffer(*reinterpret_cast<const cl::Buffer*>(src),
                                                *reinterpret_cast<cl::Buffer*>(dst), 0, 0, size, 0,
                                                0));
  } else if (kind == MemcpyKind::kHtoD) {
    CL_CHECK_OR_RETURN(queue->enqueueWriteBuffer(*reinterpret_cast<cl::Buffer*>(dst),
                                                 CL_NON_BLOCKING, 0, size, src, 0, 0));
  } else if (kind == MemcpyKind::kDtoH) {
    CL_CHECK_OR_RETURN(queue->enqueueReadBuffer(*reinterpret_cast<const cl::Buffer*>(src),
                                                CL_NON_BLOCKING, 0, size, dst, 0, 0));
  } else {
    return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}

cl_int clMemset(void* ptr, int value, size_t size) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetActiveContext(&context));
  CL_CHECK_OR_RETURN(clMemsetAsync(ptr, value, size, &(context->default_queue)));
  // blocking
  context->default_queue.finish();
  return CL_SUCCESS;
}

cl_int clMemsetAsync(void* ptr, int value, size_t size, cl::CommandQueue* queue) {
  // TODO: refactor
  if (clIsPinnedMem(ptr)) { memset(ptr, value, size); }
  std::vector<uint8_t> host_data(size, static_cast<uint8_t>(value));
  return clMemcpyAsync(ptr, host_data.data(), size, MemcpyKind::kHtoD, queue);
}

cl_int clEventCreateWithFlags(cl::Event** event, unsigned int flags) {
  *event = new cl::Event;
  return CL_SUCCESS;
}

cl_int clEventDestroy(cl::Event* event) {
  if (event) { delete event; }
  return CL_SUCCESS;
}

cl_int clEventRecord(cl::Event* event, cl::CommandQueue* queue) {
  CL_CHECK_OR_RETURN(queue->enqueueMarker(event));
  return CL_SUCCESS;
}

cl_int clEventQuery(cl::Event* event) {
  cl_int status = CL_COMPLETE;
  event->getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);
  return status;
}

cl_int clEventSynchronize(cl::Event* event) { return event->wait(); }

cl_int clQueueCreate(cl::CommandQueue** queue) {
  clContext* context = nullptr;
  CL_CHECK_OR_RETURN(clGetActiveContext(&context));
  cl_int ret = CL_SUCCESS;
  *queue = new cl::CommandQueue(context->context, context->device, 0, &ret);
  return ret;
}

cl_int clQueueDestroy(cl::CommandQueue* queue) {
  if (queue) { delete queue; }
  return CL_SUCCESS;
}

cl_int clQueueSynchronize(cl::CommandQueue* queue) { return queue->finish(); }

cl_int clQueueWaitEvent(cl::Event* event, cl::CommandQueue* queue, unsigned int flags) {
  // TODO: refactor
  return clEventSynchronize(event);
}

}  // namespace oneflow
