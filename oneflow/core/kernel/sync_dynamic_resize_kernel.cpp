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
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register_desc.h"
#include "oneflow/core/lazy/actor/actor_context.h"
#include "oneflow/core/memory/memory_case_util.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>

namespace oneflow {

#ifdef WITH_CUDA

namespace {

class CudaHostMem {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaHostMem);
  CudaHostMem(const size_t size) { OF_CUDA_CHECK(cudaMallocHost(&ptr_, size)); }
  ~CudaHostMem() { OF_CUDA_CHECK(cudaFreeHost(ptr_)); }
  void* Ptr() const { return ptr_; }

 private:
  void* ptr_;
};

}  // namespace

template<typename SizeType>
class SyncDynamicResizeGPUKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SyncDynamicResizeGPUKernel);
  SyncDynamicResizeGPUKernel() = default;
  ~SyncDynamicResizeGPUKernel() override = default;

 private:
  bool IsKernelLaunchSynchronized() const override { return false; }

  void ForwardDataContent(KernelContext* ctx) const override {
    const SyncDynamicResizeOpConf& conf = this->op_conf().sync_dynamic_resize_conf();
    CHECK_EQ(conf.axis(), 0);
    std::shared_ptr<CudaHostMem> cuda_host_mem_ptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.empty()) {
        cuda_host_mem_ptr.reset(new CudaHostMem(sizeof(SizeType)));
      } else {
        cuda_host_mem_ptr = queue_.front();
        queue_.pop();
      }
    }
    const Blob* in = ctx->BnInOp2Blob("in");
    const Blob* size = ctx->BnInOp2Blob("size");
    Blob* out = ctx->BnInOp2Blob("out");
    AutoMemcpy(ctx->stream(), out->mut_dptr(), in->dptr(), in->ByteSizeOfBlobBody(),
               out->mem_case(), in->mem_case());
    AutoMemcpy(ctx->stream(), cuda_host_mem_ptr->Ptr(), size->dptr(), sizeof(SizeType),
               memory::MakeHostMemCase(), size->mem_case());
    const auto& UpdateShape = [out, cuda_host_mem_ptr, conf, this]() {
      const int64_t new_size = *reinterpret_cast<SizeType*>(cuda_host_mem_ptr->Ptr());
      CHECK_GE(new_size, 0);
      CHECK_LE(new_size, out->shape_view().At(conf.axis()));
      // NOTE(Liang Depeng): `mut_shape_view` should be used here to get the blob's `MutShapeView`
      //                     pointer. But this callback is called after `Kernel::Forward` function's
      //                     execution and the header check is already been set to false at that
      //                     moment. So we have to choose the `ForceMutShapeView` function with
      //                     header checker disabled.
      out->ForceMutShapeView()->Set(conf.axis(), new_size);
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(cuda_host_mem_ptr);
    };
    if (conf.eager()) {
      CHECK_JUST(ctx->stream()->Sync());
      UpdateShape();
    } else {
      auto* actor_context_provider = CHECK_NOTNULL(dynamic_cast<ActorContextProvider*>(ctx));
      actor_context_provider->GetActorContext()->AddCallback(UpdateShape);
    }
  }

  mutable std::queue<std::shared_ptr<CudaHostMem>> queue_;
  mutable std::mutex mutex_;
};

#define REGISTER_SYNC_DYNAMIC_RESIZE_GPU_KERNEL(stype)                                         \
  NEW_REGISTER_KERNEL(OperatorConf::kSyncDynamicResizeConf, SyncDynamicResizeGPUKernel<stype>) \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) {                                    \
        return (kernel_conf.op_attribute().op_conf().device_tag() == "cuda"                    \
                && GetDataType<stype>::value                                                   \
                       == kernel_conf.sync_dynamic_resize_conf().size_data_type());            \
      })
REGISTER_SYNC_DYNAMIC_RESIZE_GPU_KERNEL(int8_t);
REGISTER_SYNC_DYNAMIC_RESIZE_GPU_KERNEL(int32_t);
REGISTER_SYNC_DYNAMIC_RESIZE_GPU_KERNEL(int64_t);

#endif  // WITH_CUDA

template<typename SizeType>
class SyncDynamicResizeCPUKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SyncDynamicResizeCPUKernel);
  SyncDynamicResizeCPUKernel() = default;
  ~SyncDynamicResizeCPUKernel() override = default;

 private:
  bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override {
    const SyncDynamicResizeOpConf& conf = this->op_conf().sync_dynamic_resize_conf();
    CHECK_EQ(conf.axis(), 0);
    const Blob* in = ctx->BnInOp2Blob("in");
    const Blob* size = ctx->BnInOp2Blob("size");
    Blob* out = ctx->BnInOp2Blob("out");
    AutoMemcpy(ctx->stream(), out->mut_dptr(), in->dptr(), in->ByteSizeOfBlobBody(),
               out->mem_case(), in->mem_case());
    const SizeType new_size = *size->dptr<SizeType>();
    CHECK_GE(new_size, 0);
    CHECK_LE(new_size, out->shape_view().At(conf.axis()));
    out->mut_shape_view()->Set(conf.axis(), new_size);
  }
};

#define REGISTER_SYNC_DYNAMIC_RESIZE_CPU_KERNEL(stype)                                         \
  NEW_REGISTER_KERNEL(OperatorConf::kSyncDynamicResizeConf, SyncDynamicResizeCPUKernel<stype>) \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) {                                    \
        return (kernel_conf.op_attribute().op_conf().device_tag() == "cpu"                     \
                && GetDataType<stype>::value                                                   \
                       == kernel_conf.sync_dynamic_resize_conf().size_data_type());            \
      })
REGISTER_SYNC_DYNAMIC_RESIZE_CPU_KERNEL(int8_t);
REGISTER_SYNC_DYNAMIC_RESIZE_CPU_KERNEL(int32_t);
REGISTER_SYNC_DYNAMIC_RESIZE_CPU_KERNEL(int64_t);

}  // namespace oneflow
