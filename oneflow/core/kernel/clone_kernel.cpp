#include "oneflow/core/kernel/clone_kernel.h"
#include "oneflow/core/common/meta_util.hpp"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  bool enable_fw_clone_mem_sharing =
      this->template GetValFromCustomizedOpConf<bool>("enable_fw_clone_mem_sharing");
  size_t copy_cnt = enable_fw_clone_mem_sharing ? 1 : this->op_attribute().output_bns().size();
  const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
  FOR_RANGE(size_t, i, 0, copy_cnt) {
    Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns(i));
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
  }
}

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& odbns = this->op_attribute().output_diff_bns();
  size_t out_num = odbns.size();
  if (out_num == 0) return;
  Blob* in_diff_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(0));
  auto out_diff = [&](int32_t idx) {
    return BnInOp2Blob(this->op_attribute().output_diff_bns(idx));
  };
  static const int kWidth = 8;
  int r = out_num % kWidth;
  if (r) {
    tuple_switch(r, tp_,
                 AdditionFunction<false, device_type, T, decltype(this)>{
                     in_diff_blob, std::move(BnInOp2Blob), ctx.device_ctx, 0, this});
  }
  for (; r < out_num; r += kWidth) {
    Addition<device_type, T>(ctx.device_ctx, in_diff_blob, in_diff_blob, out_diff(r),
                             out_diff(r + 1), out_diff(r + 2), out_diff(r + 3), out_diff(r + 4),
                             out_diff(r + 5), out_diff(r + 6), out_diff(r + 7));
  }
}
template<DeviceType device_type, typename T>
const PbMessage& CloneKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().clone_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCloneConf, CloneKernel, POD_DATA_TYPE_SEQ);

}  // namespace oneflow
