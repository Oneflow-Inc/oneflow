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

  const BiasAddOpConf& op_conf = this->op_conf().bias_add_conf();
  if (op_conf.axis() == a_blob->shape().NumAxes() - 1) {
    // out = bias_multiplier * b + a
    const Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), a_blob->dptr<T>(),
                        a_blob->ByteSizeOfBlobBody());
    NewKernelUtil<device_type>::OFGemm(
        ctx.device_ctx, CblasNoTrans, CblasNoTrans, bias_mul_blob->shape().elem_cnt(),
        b_blob->shape().elem_cnt(), 1, GetOneVal<T>(), bias_mul_blob->dptr<T>(), b_blob->dptr<T>(),
        GetOneVal<T>(), out_blob->mut_dptr<T>());
  } else {
    const int32_t bias_add_axis = this->op_conf().bias_add_conf().axis();
    BiasAddUtil<device_type, T>::BiasAddNCX(ctx.device_ctx, a_blob->shape(), bias_add_axis,
                                            a_blob->dptr<T>(), b_blob->dptr<T>(),
                                            out_blob->mut_dptr<T>());
  }
}

template<DeviceType device_type, typename T>
void BiasAddKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* bias_mul_blob = BnInOp2Blob("bias_multiplier");
  if (bias_mul_blob) {
    InitializerConf bias_multiplier_initializer_conf;
    bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
    NewKernelUtil<device_type>::InitializeWithConstConf(
        ctx, bias_multiplier_initializer_conf.constant_conf(), bias_mul_blob);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& BiasAddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().bias_add_conf();
}

template<typename T>
struct BiasAddUtil<DeviceType::kCPU, T> {
  static void BiasAddNCX(DeviceCtx* ctx, const Shape& shape, const int32_t bias_axis,
                         const T* input, const T* bias, T* output) {
    Shape bias_shape = Shape::Ones(shape.NumAxes());
    bias_shape.Set(bias_axis, shape.At(bias_axis));
    NdarrayUtil<DeviceType::kCPU, T>::BroadcastAdd(ctx, XpuVarNdarray<T>(shape, output),
                                                   XpuVarNdarray<const T>(shape, input),
                                                   XpuVarNdarray<const T>(bias_shape, bias));
  }
};

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kBiasAddConf, BiasAddKernel,
                                         FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
