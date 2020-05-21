#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T>
class GeluCpuKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluCpuKernel);
  GeluCpuKernel() = default;
  ~GeluCpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    int64_t n = in_blob->static_shape().elem_cnt();
    const T* x = in_blob->dptr<T>();
    T* y = BnInOp2Blob("out")->mut_dptr<T>();

    T inv_sqrt2 = std::sqrt(0.5);
    FOR_RANGE(int32_t, i, 0, n) { y[i] = 0.5 * x[i] * (1.0 + std::erf(inv_sqrt2 * x[i])); }
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().gelu_conf(); }
};

#define REGISTER_GELU_KERNELS(name, dtype)                                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::k##name##Conf, DeviceType::kCPU, dtype, \
                                        name##CpuKernel<dtype>);

REGISTER_GELU_KERNELS(Gelu, float);
REGISTER_GELU_KERNELS(Gelu, double);

template<typename T>
class GeluGradCpuKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GeluGradCpuKernel);
  GeluGradCpuKernel() = default;
  ~GeluGradCpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* x_blob = BnInOp2Blob("x");
    int64_t n = x_blob->static_shape().elem_cnt();
    const T* x = x_blob->dptr<T>();
    const T* dy = BnInOp2Blob("dy")->dptr<T>();
    T* dx = BnInOp2Blob("dx")->mut_dptr<T>();

    T inv_sqrt2 = std::sqrt(0.5);
    T coef = std::sqrt(2.0 / std::acos(-1.0));
    FOR_RANGE(int32_t, i, 0, n) {
      dx[i] = 0.5 * (1.0 + std::erf(inv_sqrt2 * x[i]) + x[i] * coef * std::exp(-0.5 * x[i] * x[i]))
              * dy[i];
    }
  }
  const PbMessage& GetCustomizedOpConf() const override { return this->op_conf().gelu_grad_conf(); }
};

REGISTER_GELU_KERNELS(GeluGrad, float);
REGISTER_GELU_KERNELS(GeluGrad, double);

#undef REGISTER_GELU_KERNELS

}  // namespace oneflow
