#ifndef ONEFLOW_CORE_KERNEL_CWISE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CWISE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class CWiseKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CWiseKernel);
  CWiseKernel() = default;
  virtual ~CWiseKernel() = default;

 private:
  void ForwardDataId(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
    BnInOp2Blob("out")->CopyDataIdFrom(ctx.device_ctx, in_blob);
  }

  void ForwardColNum(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
    BnInOp2Blob("out")->CopyColNumFrom(ctx.device_ctx, in_blob);
  }

  void BackwardColNum(const KernelCtx& ctx,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
    for (size_t i = 0; i < this->op_attribute().input_diff_bns().size(); ++i) {
      Blob* in_diff_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(i));
      in_diff_blob->CopyColNumFrom(ctx.device_ctx, out_diff_blob);
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CWISE_KERNEL_H_
