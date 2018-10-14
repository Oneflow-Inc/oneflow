#ifndef ONEFLOW_CORE_KERNEL_DEFINE_TEST_BLOB_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DEFINE_TEST_BLOB_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class DefineTestBlobKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DefineTestBlobKernel);
  DefineTestBlobKernel() = default;
  ~DefineTestBlobKernel() = default;

 private:
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataId(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  void ForwardColNum(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {}
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_DEFINE_TEST_BLOB_KERNEL_H_
