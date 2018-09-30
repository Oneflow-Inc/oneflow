#ifndef ONEFLOW_CORE_KERNEL_NCCL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NCCL_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

template<NcclUtil::NcclReduceMthd mthd>
class NcclKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclKernel);
  NcclKernel() = default;
  ~NcclKernel() override = default;

 protected:
  inline const char* InBlobName() const { return "in"; }
  inline const char* OutBlobName() const { return "out"; }

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    mthd(ctx.device_ctx, BnInOp2Blob(InBlobName()), BnInOp2Blob(OutBlobName()));
  }
};

#define NCCL_KERNEL_CLASS_NAME(mthd) Nccl##mthd##Kernel
#define DECLARE_NCCL_KERNEL(mthd)                                                \
  class NCCL_KERNEL_CLASS_NAME(mthd) final : public NcclKernel<NcclUtil::mthd> { \
   public:                                                                       \
    OF_DISALLOW_COPY_AND_MOVE(NCCL_KERNEL_CLASS_NAME(mthd));                     \
    NCCL_KERNEL_CLASS_NAME(mthd)() = default;                                    \
    ~NCCL_KERNEL_CLASS_NAME(mthd)() override = default;                          \
  };

DECLARE_NCCL_KERNEL(ReduceScatter);
DECLARE_NCCL_KERNEL(AllGather);
DECLARE_NCCL_KERNEL(AllReduce);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NCCL_KERNEL_H_
