#include "oneflow/core/kernel/instance_stack_kernel.h"
#include "oneflow/core/kernel/piece_slice_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void InstanceStackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  const size_t valid_ins_num = out_blob->dim0_valid_num(0);
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_LE(valid_ins_num, total_ins_num);
  CHECK_EQ(total_ins_num, out_blob->static_shape().At(0));
  if (ins_idx <= valid_ins_num) {
    PieceSliceKernelUtil<device_type>::InstanceStack(ctx.device_ctx, ins_idx, valid_ins_num,
                                                     BnInOp2Blob("in"), out_blob);
  }
}

template<DeviceType device_type>
void InstanceStackKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t in_diff_idx = actor_stat->first;
  const size_t valid_ins_num = out_diff_blob->dim0_valid_num(0);
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(in_diff_idx, valid_ins_num);
  CHECK_LE(valid_ins_num, total_ins_num);
  CHECK_EQ(total_ins_num, out_diff_blob->static_shape().At(0));
  if (in_diff_idx <= valid_ins_num) {
    PieceSliceKernelUtil<device_type>::PieceSlice(ctx.device_ctx, in_diff_idx, valid_ins_num,
                                                  out_diff_blob, BnInOp2Blob(GenDiffBn("in")));
  }
}

template<DeviceType device_type>
void InstanceStackKernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type>
void InstanceStackKernel<device_type>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

template<DeviceType device_type>
void InstanceStackKernel<device_type>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // TODO
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kInstanceStackConf, InstanceStackKernel);

}  // namespace oneflow
