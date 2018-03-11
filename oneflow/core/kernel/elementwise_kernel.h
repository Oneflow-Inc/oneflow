#ifndef ONEFLOW_CORE_KERNEL_ELEMENTWISE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ELEMENTWISE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class ElementwiseKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ElementwiseKernel);
  ElementwiseKernel() = default;
  virtual ~ElementwiseKernel() = default;

 private:
  void ForwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* in_blob = BnInOp2Blob(this->kernel_conf().input_bns()[0]);
    BnInOp2Blob("out")->CopyDataIdFrom(ctx.device_ctx, in_blob);
  }

  void ForwardColNum(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* in_blob = BnInOp2Blob(this->kernel_conf().input_bns()[0]);
    BnInOp2Blob("out")->CopyColNumFrom(ctx.device_ctx, in_blob);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_Add_KERNEL_H_
