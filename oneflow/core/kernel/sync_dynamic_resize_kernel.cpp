#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

namespace {

class CudaHostMem {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaHostMem);
  CudaHostMem(const size_t size) { CudaCheck(cudaMallocHost(&ptr_, size)); }
  ~CudaHostMem() { CudaCheck(cudaFreeHost(ptr_)); }
  void* Ptr() const { return ptr_; }

 private:
  void* ptr_;
};

}  // namespace

class SyncDynamicResizeKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SyncDynamicResizeKernel);
  SyncDynamicResizeKernel() = default;
  ~SyncDynamicResizeKernel() override = default;

 private:
  bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const SyncDynamicResizeOpConf& conf = this->op_conf().sync_dynamic_resize_conf();
    CHECK_EQ(conf.axis(), 0);
    std::shared_ptr<CudaHostMem> cuda_host_mem_ptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.empty()) {
        cuda_host_mem_ptr.reset(new CudaHostMem(sizeof(int32_t)));
      } else {
        cuda_host_mem_ptr = queue_.front();
        queue_.pop();
      }
    }
    const Blob* in = BnInOp2Blob("in");
    const Blob* size = BnInOp2Blob("size");
    Blob* out = BnInOp2Blob("out");
    AutoMemcpy(ctx.device_ctx, out->mut_dptr(), in->dptr(), in->ByteSizeOfBlobBody(),
               out->mem_case(), in->mem_case());
    AutoMemcpy(ctx.device_ctx, cuda_host_mem_ptr->Ptr(), size->dptr(), sizeof(int32_t),
               MakeHostMemCase(), size->mem_case());
    ctx.device_ctx->AddCallBack([out, cuda_host_mem_ptr, conf, this]() {
      const int64_t new_size = *reinterpret_cast<int32_t*>(cuda_host_mem_ptr->Ptr());
      CHECK_GE(new_size, 0);
      CHECK_LE(new_size, out->shape_view().At(conf.axis()));
      out->mut_shape_view()->Set(conf.axis(), new_size);
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(cuda_host_mem_ptr);
    });
  }

  mutable std::queue<std::shared_ptr<CudaHostMem>> queue_;
  mutable std::mutex mutex_;
};

REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kSyncDynamicResizeConf, SyncDynamicResizeKernel);

}  // namespace oneflow
