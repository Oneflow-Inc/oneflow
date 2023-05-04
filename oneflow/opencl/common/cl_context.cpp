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
#include "oneflow/opencl/common/cl_context.h"

#include "oneflow/opencl/common/cl_util.h"

namespace oneflow {

clContextPool::clContextPool() {
  std::vector<cl::Platform> platforms;
  OF_CL_CHECK(cl::Platform::get(&platforms));
  CHECK_GT(platforms.size(), 0);
  platform_ = platforms[0];
  OF_CL_CHECK(platform_.getDevices(CL_DEVICE_TYPE_GPU, &devices_));
}

/*static*/ clContextPool* clContextPool::Get() {
  static clContextPool context_pool;
  return &context_pool;
}

cl_int clContextPool::getOrCreateContext(clContext** context, int device_id) {
  if (device_id < 0 || device_id >= devices_.size()) { return CL_DEVICE_NOT_AVAILABLE; }
  cl_int ret = CL_SUCCESS;
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = contexts_.find(device_id);
  if (it == contexts_.end()) {
    std::unique_ptr<clContext> context(new clContext);
    context->device_id = device_id;
    context->device = devices_[device_id];
    context->context = cl::Context(context->device, 0, 0, 0, &ret);
    if (ret != CL_SUCCESS) { return ret; }
    context->default_queue = cl::CommandQueue(context->context, devices_[device_id], 0, &ret);
    if (ret != CL_SUCCESS) { return ret; }
    it = contexts_.emplace(device_id, std::move(context)).first;
  }
  *context = it->second.get();
  return ret;
}

cl_int clContextPool::getDevices(cl::Device** devices, int* device_count) {
  if (devices) { *devices = devices_.data(); }
  *device_count = devices_.size();
  return CL_SUCCESS;
}

}  // namespace oneflow
