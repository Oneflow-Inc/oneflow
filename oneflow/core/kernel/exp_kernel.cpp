#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ExpKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ExpKernel);
  ExpKernel() = default;
  ~ExpKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* in_blob = BnInOp2Blob("in");
    NewKernelUtil<device_type>::Exp(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                    BnInOp2Blob("out")->mut_dptr<T>());
  }
};

#define REGISTER_EXP_KERNEL(dev, dtype) \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kExpConf, dev, dtype, ExpKernel<dev, dtype>)
REGISTER_EXP_KERNEL(DeviceType::kGPU, float);
REGISTER_EXP_KERNEL(DeviceType::kGPU, double);
REGISTER_EXP_KERNEL(DeviceType::kCPU, float);
REGISTER_EXP_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
