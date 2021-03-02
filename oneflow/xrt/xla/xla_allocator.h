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
#ifndef ONEFLOW_XRT_XLA_XLA_ALLOCATOR_H_
#define ONEFLOW_XRT_XLA_XLA_ALLOCATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/xrt/xla/memory/device_buffer_allocator.h"

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace oneflow {
namespace xrt {
namespace mola {

namespace se = tensorflow::se;
using uint64 = tensorflow::uint64;
using int64 = tensorflow::int64;

class XlaAllocator : public se::DeviceMemoryAllocator {
 public:
  explicit XlaAllocator(const se::Platform *platform, DeviceBufferAllocator *allocator);
  virtual ~XlaAllocator();
  using se::DeviceMemoryAllocator::Allocate;
  xla::StatusOr<se::OwningDeviceMemory> Allocate(int device_ordinal, uint64 size,
                                                 bool retry_on_failure,
                                                 int64 /*memory_space*/) override;
  tensorflow::Status Deallocate(int device_ordinal, se::DeviceMemoryBase mem) override;

  bool AllowsAsynchronousDeallocation() const override { return true; }

  void ResetState();
  void ReserveWorkspace(size_t workspace_bytes);
  void LockWorkspace();
  void UnlockWorkspace();

  void PopulateDeviceMemory(const std::vector<se::DeviceMemoryBase> &device_buffers,
                            const std::vector<int64_t> &allocation_indices);
  stream_executor::port::StatusOr<stream_executor::Stream *> GetStream(
      int device_ordinal) override {
    UNIMPLEMENTED();
  };

 private:
  DeviceBufferAllocator *allocator_;
  int64_t allocate_offset_;
  int64_t allocate_index_;

  struct AllocationBuffer {
    bool populated = false;
    int64_t index = -1;
    se::DeviceMemoryBase memory;
  };
  std::vector<AllocationBuffer> populated_buffers_;
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_ALLOCATOR_H_
