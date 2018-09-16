#include "oneflow/core/kernel/shared_model_diff_add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void SharedModelDiffAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& ibns = this->op_attribute().input_bns();
  size_t in_num = ibns.size();
  if (in_num == 0) return;
  Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
  auto in_blob = [&](int32_t idx) { return BnInOp2Blob(this->op_attribute().input_bns(idx)); };
  static const int kWidth = 8;
  int r = in_num % kWidth;
  if (r) {
    tuple_switch(r, tp_,
                 AdditionFunction<true, device_type, T, decltype(this)>{
                     out_blob, std::move(BnInOp2Blob), ctx.device_ctx, 0, this});
  }
  for (; r < in_num; r += kWidth) {
    Addition<device_type, T>(ctx.device_ctx, out_blob, out_blob, in_blob(r), in_blob(r + 1),
                             in_blob(r + 2), in_blob(r + 3), in_blob(r + 4), in_blob(r + 5),
                             in_blob(r + 6), in_blob(r + 7));
  }
}

template<DeviceType device_type, typename T>
const PbMessage& SharedModelDiffAddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().shared_model_diff_add_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSharedModelDiffAddConf, SharedModelDiffAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
