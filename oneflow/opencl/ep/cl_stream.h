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
#ifndef ONEFLOW_OPENCL_EP_CL_STREAM_H_
#define ONEFLOW_OPENCL_EP_CL_STREAM_H_

#include "oneflow/opencl/ep/cl_device.h"
#include "oneflow/opencl/common/cl_util.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/vm/caching_allocator.h"

namespace oneflow {
namespace ep {

class clDevice;

class clStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(clStream);
  explicit clStream(clDevice* device);
  ~clStream() override;

  static constexpr uint32_t kDefaultBlockSize = 256;

  DeviceType device_type() const override;

  clDevice* device() const override;

  void RecordEvent(Event* event) override;
  void WaitEvent(Event* event) override;

  Maybe<void> Sync() override;
  Maybe<void> GetAsyncError() override;

  Maybe<void> OnExecutionContextSetup() override;
  Maybe<void> OnExecutionContextTeardown() override;

  cl::CommandQueue* cl_stream() const;

 private:
  cl::CommandQueue* cl_stream_;
  int device_index_;
  clDevice* device_;
};

}  // namespace ep
}  // namespace oneflow

#endif  // ONEFLOW_OPENCL_EP_CL_STREAM_H_
