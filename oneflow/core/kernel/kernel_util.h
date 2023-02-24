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
#ifndef ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_

#include "oneflow/core/common/blas.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

class Blob;
class MemoryCase;
class StreamContext;

void AutoMemcpy(ep::Stream* stream, void* dst, const void* src, size_t sz,
                const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case);
void AutoMemcpy(ep::Stream* stream, Blob* dst, const Blob* src);
void SyncAutoMemcpy(ep::Stream* stream, void* dst, const void* src, size_t sz,
                    const MemoryCase& dst_mem_case, const MemoryCase& src_mem_case);
void AutoMemset(ep::Stream* stream, void* dst, const char value, size_t sz,
                const MemoryCase& dst_mem_case);
namespace {
class PinnedMemoryGuard {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PinnedMemoryGuard);
  PinnedMemoryGuard(ep::Device* device, size_t size) : device_(device) {
    options_.SetPinnedDevice(device->device_type(), 0);
    CHECK_JUST(device_->AllocPinned(options_, &ptr_, size));
  }

  ~PinnedMemoryGuard() { device_->FreePinned(options_, ptr_); }

  template<typename T = void>
  T* ptr() {
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  ep::AllocationOptions options_;
  ep::Device* device_;
  void* ptr_{};
};

}  // namespace

template<typename T>
void DebugTensor(user_op::KernelComputeContext* ctx, const user_op::Tensor* t,
                 const std::string& debug_prefix) {
  std::unique_ptr<ep::primitive::Memcpy> d2h =
      ep::primitive::NewPrimitive<ep::primitive::MemcpyFactory>(ctx->device_type(),
                                                                ep::primitive::MemcpyKind::kDtoH);
  int64_t t_size = t->shape_view().elem_cnt() * GetSizeOfDataType(t->data_type());
  auto device = Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(
      DeviceType::kCUDA, ctx->parallel_ctx().parallel_id());
  CHECK(device);
  PinnedMemoryGuard output(device.get(), t_size);
  d2h->Launch(ctx->stream(), output.ptr(), t->dptr(), t_size);
  CHECK_JUST(ctx->stream()->Sync());
  auto* outptr = output.ptr<T>();
  LOG(INFO) << debug_prefix << " datatype " << t->data_type() << " cur parallel id "
            << ctx->parallel_ctx().parallel_id() << " ptr " << t->dptr();
  for (int i = 0; i < t->shape_view().elem_cnt(); ++i) {
    LOG(INFO) << " index " << i << ", value " << *(outptr + i);
  }
  google::FlushLogFiles(google::INFO);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
