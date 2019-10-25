#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MultiplyKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiplyKernel);
  MultiplyKernel() = default;
  ~MultiplyKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_0_blob = BnInOp2Blob("in_0");
    const Blob* in_1_blob = BnInOp2Blob("in_1");
    Blob* out_blob = BnInOp2Blob("out");
    // out = in_0 .* in_1
    NewKernelUtil<device_type>::Mul(ctx.device_ctx, in_0_blob->shape().elem_cnt(),
                                    in_0_blob->dptr<T>(), in_1_blob->dptr<T>(),
                                    out_blob->mut_dptr<T>());
  };
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().multiply_conf(); }
};

#define REGISTER_ARITHMETIC_KERNEL(name, dev, dtype)                             \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::k##name##Conf, dev, dtype, \
                                        name##Kernel<dev, dtype>);

#define REGISTER_WITH_NAME_AND_DTYPE(name, dtype)           \
  REGISTER_ARITHMETIC_KERNEL(name, DeviceType::kCPU, dtype) \
  REGISTER_ARITHMETIC_KERNEL(name, DeviceType::kGPU, dtype)

REGISTER_WITH_NAME_AND_DTYPE(Multiply, float);
REGISTER_WITH_NAME_AND_DTYPE(Multiply, double);

#undef REGISTER_WITH_NAME_AND_DTYPE
#undef REGISTER_ARITHMETIC_KERNEL

}  // namespace oneflow
