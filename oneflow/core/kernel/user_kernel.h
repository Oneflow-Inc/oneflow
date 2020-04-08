#ifndef ONEFLOW_CORE_KERNEL_USER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_USER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

class UserKernelComputeContext;

class UserKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserKernel);
  UserKernel() : Kernel(), ctx_(nullptr) {}
  ~UserKernel();

  // for eager mode
  UserKernel(const JobDesc* job_desc, const KernelConf& kernel_conf);
  // return new OpKernelState
  std::shared_ptr<user_op::OpKernelState> EagerModelForward(
      const std::shared_ptr<user_op::OpKernelState>& old_opkernel_state, DeviceCtx* device_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob);

 private:
  void InitOpKernel();
  void InitComputeContext(DeviceCtx* device_ctx);
  void InitUserKernel(DeviceCtx* device_ctx);
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(DeviceCtx* device_ctx);
  void VirtualKernelInit(DeviceCtx* device_ctx) override;
  void ForwardUserKernel(std::function<Blob*(const std::string&)> BnInOp2Blob,
                         user_op::OpKernelState* opkernel_state) const;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::unique_ptr<UserKernelComputeContext> ctx_;
  std::shared_ptr<user_op::OpKernelState> opkernel_state_;
  std::unique_ptr<const user_op::OpKernel> kernel_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_USER_KERNEL_H_
