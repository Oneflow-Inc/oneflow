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
#ifndef ONEFLOW_CORE_EP_NPU_NPU_STREAM_H_
#define ONEFLOW_CORE_EP_NPU_NPU_STREAM_H_


#ifdef WITH_NPU
#include "oneflow/core/ep/include/stream.h"
#include "acl/acl.h"
#include "acl/acl_base.h"

#include "oneflow/core/device/npu_util.h"

namespace oneflow {

namespace ep {

class NpuDevice;

class NpuStream : public Stream {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NpuStream);
  explicit NpuStream(NpuDevice* device);
  ~NpuStream() override;

  static constexpr uint32_t kDefaultBlockSize = 256;

  DeviceType device_type() const override;
  Device* device() const override;
  Maybe<void> Sync() override;
  void RecordEvent(Event* event) override;

  Maybe<void> OnExecutionContextSetup() override;
  Maybe<void> OnExecutionContextTeardown() override;

  aclrtStream npu_stream() const;

 private:
  aclrtStream npu_stream_{};

  int device_index_;
  void* workspace_{};
  size_t workspace_size_{};

  NpuDevice* device_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_NPU

#endif  // ONEFLOW_CORE_EP_NPU_NPU_STREAM_H_
