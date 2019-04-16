#include "oneflow/core/kernel/tuple_identity_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void TupleIdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& input_bns = this->op_attribute().input_bns();
  const auto& output_bns = this->op_attribute().output_bns();
  CHECK_EQ(input_bns.size(), output_bns.size());
  FOR_RANGE(int, i, 0, input_bns.size()) {
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    out_blob->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(input_bns.Get(i)));
  }
}

template<DeviceType device_type>
void TupleIdentityKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& input_diff_bns = this->op_attribute().input_diff_bns();
  const auto& output_diff_bns = this->op_attribute().output_diff_bns();
  CHECK_EQ(input_diff_bns.size(), output_diff_bns.size());
  FOR_RANGE(int, i, 0, output_diff_bns.size()) {
    Blob* in_diff_blob = BnInOp2Blob(input_diff_bns.Get(i));
    in_diff_blob->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob(output_diff_bns.Get(i)));
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kTupleIdentityConf, TupleIdentityKernel);
ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kTupleBufferConf, TupleIdentityKernel);

}  // namespace oneflow
