#include "oneflow/core/kernel/add_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_common.hpp"

namespace oneflow {

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& ibns = this->op_attribute().input_bns();
  size_t in_num = ibns.size();
  if (in_num == 0) return;
  Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns(0));
  Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), 0,
                      out_blob->ByteSizeOfDataContentField());
  auto in_blob = [&](int32_t idx) { return BnInOp2Blob(this->op_attribute().input_bns(idx)); };
  int32_t offset = 0;
  while (in_num - offset >= 10) {
    AdditionAssign<device_type, T>(ctx.device_ctx, out_blob, in_blob(offset), in_blob(offset + 1),
                                   in_blob(offset + 2), in_blob(offset + 3), in_blob(offset + 4),
                                   in_blob(offset + 5), in_blob(offset + 6), in_blob(offset + 7),
                                   in_blob(offset + 8), in_blob(offset + 9));
    offset += 10;
  }

  if (in_num - offset > 0) {
    tuple_switch(in_num - offset, tp_,
                 KernelFunction<true, device_type, T, decltype(this)>{
                     out_blob, std::move(BnInOp2Blob), ctx.device_ctx, offset, this});
  }
}

template<DeviceType device_type, typename T>
void AddKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  FOR_RANGE(size_t, i, 0, this->op_attribute().input_diff_bns().size()) {
    Blob* in_diff_blob = BnInOp2Blob(this->op_attribute().input_diff_bns(i));
    in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& AddKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().add_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAddConf, AddKernel, ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
