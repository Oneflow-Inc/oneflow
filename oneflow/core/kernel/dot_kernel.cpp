#include "oneflow/core/kernel/dot_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void DotKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const int64_t piece_size = in_blob->shape().At(0);
  const int64_t dim = in_blob->shape().Count(1);
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* tmp_blob = BnInOp2Blob("tmp");
  Blob* tmp_storage_blob = BnInOp2Blob("tmp_storage");
  // out = in .* weight
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, piece_size * dim, in_blob->dptr<T>(),
                                  weight_blob->dptr<T>(), tmp_blob->mut_dptr<T>());
  KernelUtil<device_type, T>::RowSum(ctx.device_ctx, piece_size, dim, tmp_blob->dptr<T>(),
                                     out_blob->mut_dptr<T>(), tmp_storage_blob->mut_dptr<T>(),
                                     sizeof(T) * piece_size * dim);
  if (this->op_conf().dot_conf().has_bias()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    // out += bias
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, piece_size, GetOneVal<T>(),
                                     bias_blob->dptr<T>(), 1, out_blob->mut_dptr<T>(), 1);
  }
}

template<DeviceType device_type, typename T>
void DotKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf diff_multiplier_initializer_conf;
  diff_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
  KernelUtil<device_type, T>::InitializeWithConf(ctx, diff_multiplier_initializer_conf, 0,
                                                 BnInOp2Blob("diff_multiplier"));
}

template<DeviceType device_type, typename T>
const PbMessage& DotKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().dot_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDotConf, DotKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
