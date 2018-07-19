#include "oneflow/core/kernel/clone_kernel.h"
#include "oneflow/core/common/meta_util.hpp"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::Forward(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns(0));
  for (const std::string& obn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_memory_ptr(), in_blob->memory_ptr(),
                        in_blob->TotalByteSize());
  }
}

template<DeviceType device_type, typename T>
void CloneKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& odbns = this->op_attribute().output_diff_bns();
  size_t out_num = odbns.size();
  if (out_num == 0) return;
  Blob* in_diff_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(0));
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  auto out_diff = [&](int32_t idx) {
    return BnInOp2Blob(this->op_attribute().output_diff_bns(idx));
  };
  int32_t offset = 0;
  while (out_num - offset >= 10) {
    AdditionAssign<device_type, T>(
        ctx.device_ctx, in_diff_blob, out_diff(offset), out_diff(offset + 1), out_diff(offset + 2),
        out_diff(offset + 3), out_diff(offset + 4), out_diff(offset + 5), out_diff(offset + 6),
        out_diff(offset + 7), out_diff(offset + 8), out_diff(offset + 9));
    offset += 10;
  }

  if (out_num - offset > 0) {
    tuple_switch(out_num - offset, tp_,
                 AdditionAssignFunction<false, device_type, T, decltype(this)>{
                     in_diff_blob, std::move(BnInOp2Blob), ctx.device_ctx, offset, this});
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCloneConf, CloneKernel, POD_DATA_TYPE_SEQ);

}  // namespace oneflow
