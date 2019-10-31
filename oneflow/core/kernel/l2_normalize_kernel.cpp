#include "oneflow/core/kernel/l2_normalize_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class L2NormalizeKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(L2NormalizeKernel);
  L2NormalizeKernel() = default;
  ~L2NormalizeKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    L2NormalizeKernelUtil<device_type, T>::Forward(
        ctx.device_ctx, this->op_conf().l2_normalize_conf().axis(),
        this->op_conf().l2_normalize_conf().epsilon(), BnInOp2Blob("in"),
        BnInOp2Blob("square_x_sum"), BnInOp2Blob("out"));
  }
};

#define REGISTER_L2_NORMALIZE_KERNEL(dev, dtype)                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kL2NormalizeConf, dev, dtype, \
                                        L2NormalizeKernel<dev, dtype>)
REGISTER_L2_NORMALIZE_KERNEL(DeviceType::kGPU, float);
REGISTER_L2_NORMALIZE_KERNEL(DeviceType::kGPU, double);
REGISTER_L2_NORMALIZE_KERNEL(DeviceType::kCPU, float);
REGISTER_L2_NORMALIZE_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
