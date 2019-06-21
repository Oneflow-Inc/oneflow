#include "oneflow/core/kernel/gelu_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void GeluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  GeluKernelUtil<device_type, T>::GeluForward(ctx.device_ctx, in_blob->static_shape().elem_cnt(),
                                              in_blob->dptr<T>(),
                                              BnInOp2Blob("out")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
const PbMessage& GeluKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gelu_conf();
}

template<typename T>
struct GeluKernelUtil<DeviceType::kCPU, T> {
  static void GeluForward(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    T inv_sqrt2 = std::sqrt(0.5);
    FOR_RANGE(int32_t, i, 0, n) { y[i] = 0.5 * x[i] * (1.0 + std::erf(inv_sqrt2 * x[i])); }
  }

  static void GeluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* dy, T* dx) {
    T inv_sqrt2 = std::sqrt(0.5);
    T coef = std::sqrt(2.0 / std::acos(-1.0));
    FOR_RANGE(int32_t, i, 0, n) {
      dx[i] = 0.5 * (1.0 + std::erf(inv_sqrt2 * x[i]) + x[i] * coef * std::exp(-0.5 * x[i] * x[i]))
              * dy[i];
    }
  }
};

#define INSTANTIATE_GELU_KERNEL_UTIL(type_cpp, type_proto) \
  template struct GeluKernelUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GELU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGeluConf, GeluKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
