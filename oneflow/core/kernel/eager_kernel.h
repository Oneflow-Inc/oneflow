#ifndef ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

class EagerKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerKernel);
  EagerKernel(const JobDesc* job_desc, const KernelConf& kernel_conf);
  ~EagerKernel() = default;

  void Infer(std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  std::shared_ptr<user_op::OpKernelState> EagerModelForward(
      const std::shared_ptr<user_op::OpKernelState>& old_opkernel_state, DeviceCtx* device_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

 private:
  void InitOpKernel(const KernelConf& kernel_conf);
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    UNIMPLEMENTED();
  }
  std::unique_ptr<const user_op::OpKernel> kernel_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_EAGER_KERNEL_H_
