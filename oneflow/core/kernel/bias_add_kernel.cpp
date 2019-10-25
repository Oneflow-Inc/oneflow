#include "oneflow/core/kernel/bias_add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void BiasAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a_blob = BnInOp2Blob("a");
  const Blob* b_blob = BnInOp2Blob("b");
  Blob* out_blob = BnInOp2Blob("out");
  const BiasAddOpConf& conf = this->op_conf().bias_add_conf();
  const int32_t bias_add_axis = conf.axis();
  const int64_t outer_size = a_blob->shape().Count(0, bias_add_axis);
  const int64_t bias_size = a_blob->shape().At(bias_add_axis);
  const int64_t inner_size = a_blob->shape().Count(bias_add_axis + 1);
  BiasAddUtil<device_type, T>::BiasAdd(ctx.device_ctx, outer_size, bias_size, inner_size,
                                       a_blob->dptr<T>(), b_blob->dptr<T>(),
                                       out_blob->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
const PbMessage& BiasAddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().bias_add_conf();
}

template<typename T>
struct BiasAddUtil<DeviceType::kCPU, T> {
  static void BiasAdd(DeviceCtx* ctx, int64_t outer_size, int64_t bias_size, int64_t inner_size,
                      const T* x, const T* bias, T* y) {
    const Shape in_out_shape({outer_size, bias_size, inner_size});
    const Shape bias_shape({1, bias_size, 1});
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastAdd(ctx, XpuVarNdarray<T>(in_out_shape, y),
                                                   XpuVarNdarray<const T>(in_out_shape, x),
                                                   XpuVarNdarray<const T>(bias_shape, bias));
  }
};

#define INSTANTIATE_BIAS_ADD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct BiasAddUtil<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_BIAS_ADD_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ)

#define REGISTER_BIAS_ADD_KERNEL(dev, dtype)                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBiasAddConf, dev, dtype, \
                                        BiasAddKernel<dev, dtype>)

REGISTER_BIAS_ADD_KERNEL(DeviceType::kGPU, float);
REGISTER_BIAS_ADD_KERNEL(DeviceType::kGPU, double);
REGISTER_BIAS_ADD_KERNEL(DeviceType::kGPU, int8_t);
REGISTER_BIAS_ADD_KERNEL(DeviceType::kGPU, int32_t);
REGISTER_BIAS_ADD_KERNEL(DeviceType::kGPU, int64_t);

REGISTER_BIAS_ADD_KERNEL(DeviceType::kCPU, float);
REGISTER_BIAS_ADD_KERNEL(DeviceType::kCPU, double);
REGISTER_BIAS_ADD_KERNEL(DeviceType::kCPU, int8_t);
REGISTER_BIAS_ADD_KERNEL(DeviceType::kCPU, int32_t);
REGISTER_BIAS_ADD_KERNEL(DeviceType::kCPU, int64_t);

}  // namespace oneflow
