#include "oneflow/core/kernel/multiply_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template <DeviceType device_type, typename T>
const PbMessage& MultiplyGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().segment_sum_conf();
}

template <DeviceType device_type, typename T>
void MultiplyGradKernel<device_type, T>::ForwardDataContent(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* in_0_blob = BnInOp2Blob("in_0");
  Blob* in_0_diff_blob = BnInOp2Blob("in_0_diff");
  const Blob* in_1_blob = BnInOp2Blob("in_1");
  Blob* in_1_diff_blob = BnInOp2Blob("in_1_diff");
  // in_1_diff = out_diff * in_0
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, in_0_blob->shape().elem_cnt(),
                                  in_0_blob->dptr<T>(), out_diff_blob->dptr<T>(),
                                  in_1_diff_blob->mut_dptr<T>());
  // in_0_diff = out_diff * in_1
  KernelUtil<device_type, T>::Mul(ctx.device_ctx, in_1_blob->shape().elem_cnt(),
                                  in_1_blob->dptr<T>(), out_diff_blob->dptr<T>(),
                                  in_0_diff_blob->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMultiplyGradConf, MultiplyGradKernel, FLOATING_DATA_TYPE_SEQ);

} // namespace oneflow
