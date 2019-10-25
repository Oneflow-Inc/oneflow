#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ScalarMulKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarMulKernel);
  ScalarMulKernel() = default;
  ~ScalarMulKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    T scalar_operand = static_cast<T>(0);
    const auto& conf = this->op_conf().scalar_mul_conf();
    if (conf.has_int_operand()) {
      scalar_operand = static_cast<T>(conf.int_operand());
    } else if (conf.has_float_operand()) {
      scalar_operand = static_cast<T>(conf.float_operand());
    } else {
      UNIMPLEMENTED();
    }
    NewKernelUtil<device_type>::MulByScalar(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                            in_blob->dptr<T>(), scalar_operand,
                                            out_blob->mut_dptr<T>());
  }
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().scalar_mul_conf();
  }
};

#define REGISTER_SCALAR_ARITHMETIC_KERNEL(name, dev, dtype)                            \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kScalar##name##Conf, dev, dtype, \
                                        Scalar##name##Kernel<dev, dtype>);

#define REGISTER_WITH_NAME_AND_DTYPE(name, dtype)                  \
  REGISTER_SCALAR_ARITHMETIC_KERNEL(name, DeviceType::kCPU, dtype) \
  REGISTER_SCALAR_ARITHMETIC_KERNEL(name, DeviceType::kGPU, dtype)

REGISTER_WITH_NAME_AND_DTYPE(Mul, float);
REGISTER_WITH_NAME_AND_DTYPE(Mul, double);
REGISTER_WITH_NAME_AND_DTYPE(Mul, int32_t);

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kScalarMulConf, DeviceType::kGPU, float16,
                                      ScalarMulKernel<DeviceType::kGPU, float16>);

template<DeviceType device_type, typename T>
class ScalarAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarAddKernel);
  ScalarAddKernel() = default;
  ~ScalarAddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    T scalar_operand = static_cast<T>(0);
    const auto& conf = this->op_conf().scalar_add_conf();
    if (conf.has_int_operand()) {
      scalar_operand = static_cast<T>(conf.int_operand());
    } else if (conf.has_float_operand()) {
      scalar_operand = static_cast<T>(conf.float_operand());
    } else {
      UNIMPLEMENTED();
    }
    NewKernelUtil<device_type>::AddByScalar(ctx.device_ctx, out_blob->shape().elem_cnt(),
                                            in_blob->dptr<T>(), scalar_operand,
                                            out_blob->mut_dptr<T>());
  }
  const PbMessage& GetCustomizedOpConf() const override {
    return this->op_conf().scalar_add_conf();
  }
};

REGISTER_WITH_NAME_AND_DTYPE(Add, float);
REGISTER_WITH_NAME_AND_DTYPE(Add, double);
REGISTER_WITH_NAME_AND_DTYPE(Add, int32_t);

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kScalarAddConf, DeviceType::kGPU, float16,
                                      ScalarAddKernel<DeviceType::kGPU, float16>);

#undef REGISTER_WITH_NAME_AND_DTYPE

}  // namespace oneflow
