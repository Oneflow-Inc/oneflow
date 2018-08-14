#include "oneflow/core/kernel/dot_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void DotKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  const int64_t dim = in_blob->shape().Count(1);
  const int64_t piece_size = in_blob->shape().At(0);
  // out = in .* weight
  FOR_RANGE(int64_t, i, 0, piece_size) {
    KernelUtil<device_type, T>::Dot(ctx.device_ctx, dim, in_blob->dptr<T>() + i * dim, 1,
                                    weight_blob->dptr<T>() + i * dim, 1,
                                    out_blob->mut_dptr<T>() + i);
  }
  if (this->op_conf().matmul_conf().has_bias()) {
    const Blob* bias_blob = BnInOp2Blob("bias");
    // out += bias
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, piece_size, OneVal<T>::value,
                                     bias_blob->dptr<T>(), 1, out_blob->mut_dptr<T>(), 1);
  }
}

template<DeviceType device_type, typename T>
void DotKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  const int64_t dim = in_blob->shape().Count(1);
  const int64_t piece_size = in_blob->shape().At(0);
  // weight_diff = out_diff * in
  FOR_RANGE(int64_t, i, 0, piece_size) {
    KernelUtil<device_type, T>::Copy(ctx.device_ctx, dim, in_blob->dptr<T>() + i * dim, 1,
                                     weight_diff_blob->mut_dptr<T>() + i * dim, 1);
    KernelUtil<device_type, T>::Scal(ctx.device_ctx, dim, out_diff_blob->dptr<T>()[i],
                                     weight_diff_blob->mut_dptr<T>() + i * dim, 1);
  }
  // in_diff = out_diff * weight
  FOR_RANGE(int64_t, i, 0, piece_size) {
    KernelUtil<device_type, T>::Copy(ctx.device_ctx, dim, weight_blob->dptr<T>() + i * dim, 1,
                                     in_diff_blob->mut_dptr<T>() + i * dim, 1);
    KernelUtil<device_type, T>::Scal(ctx.device_ctx, dim, out_diff_blob->dptr<T>()[i],
                                     in_diff_blob->mut_dptr<T>() + i * dim, 1);
  }
  if (this->op_conf().matmul_conf().has_bias()) {
    Blob* bias_diff_blob = BnInOp2Blob("bias_diff");
    // bias_diff = out_diff
    KernelUtil<device_type, T>::Copy(ctx.device_ctx, piece_size, out_diff_blob->dptr<T>(), 1,
                                     bias_diff_blob->mut_dptr<T>(), 1);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& DotKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().dot_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDotConf, DotKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
