#include "oneflow/core/kernel/recurrent_kernel.h"
#include "oneflow/core/kernel/basic_rnn_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void RecurrentKernel<device_type, T>::VirtualKernelInit(const ParallelContext*) {
  auto& input_bns = this->op_attribute().input_bns();
  need_external_h0_ = std::find(input_bns.begin(), input_bns.end(), "h0") != input_bns.end();
}

template<DeviceType device_type, typename T>
bool RecurrentKernel<device_type, T>::NeedExternalH0() const {
  return need_external_h0_;
}

template<DeviceType device_type, typename T>
Blob* RecurrentKernel<device_type, T>::GetHiddenBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0"); }
  return BnInOp2Blob("rec_in");
}

template<DeviceType device_type, typename T>
Blob* RecurrentKernel<device_type, T>::GetHiddenDiffBlob(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (BnInOp2Blob("in")->col_id() == 0) { return BnInOp2Blob("h0_diff"); }
  return BnInOp2Blob("rec_in_diff");
}

template<DeviceType device_type, typename T>
void RecurrentKernel<device_type, T>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type, typename T>
void RecurrentKernel<device_type, T>::ForwardColNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
  BnInOp2Blob("rec_out")->CopyColNumFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

#define INSTANTIATE_KERNEL(device_type, data_type_pair) \
  template class RecurrentKernel<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_KERNEL, DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
