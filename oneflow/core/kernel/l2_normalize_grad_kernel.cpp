#include "oneflow/core/kernel/l2_normalize_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class L2NormalizeGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeGradKernel);
  L2NormalizeGradKernel() = default;
  ~L2NormalizeGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    L2NormalizeKernelUtil<device_type, T>::Backward(
        ctx.device_ctx, this->op_conf().l2_normalize_grad_conf().axis(),
        this->op_conf().l2_normalize_grad_conf().epsilon(), BnInOp2Blob("y"), BnInOp2Blob("dy"),
        BnInOp2Blob("square_x_sum"), BnInOp2Blob("dx"));
  }
};

#define REGISTER_L2_NORMALIZE_GRAD_KERNEL(dev, dtype)                                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kL2NormalizeGradConf, dev, dtype, \
                                        L2NormalizeGradKernel<dev, dtype>)
REGISTER_L2_NORMALIZE_GRAD_KERNEL(DeviceType::kGPU, float);
REGISTER_L2_NORMALIZE_GRAD_KERNEL(DeviceType::kGPU, double);
REGISTER_L2_NORMALIZE_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_L2_NORMALIZE_GRAD_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
