#include "oneflow/core/kernel/instance_stack_kernel.h"
#include "oneflow/core/kernel/piece_slice_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void InstanceStackKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  is_first_in_ = true;
}

template<DeviceType device_type, typename T>
void InstanceStackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_EQ(total_ins_num, out_blob->static_shape().At(0));
  PieceSliceKernelUtil<device_type, T>::InstanceStack(ctx.device_ctx, ins_idx, total_ins_num,
                                                      BnInOp2Blob("in"), out_blob);
}

template<DeviceType device_type, typename T>
void InstanceStackKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t in_diff_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(in_diff_idx, total_ins_num);
  CHECK_EQ(total_ins_num, out_diff_blob->static_shape().At(0));
  PieceSliceKernelUtil<device_type, T>::PieceSlice(ctx.device_ctx, in_diff_idx, total_ins_num,
                                                   out_diff_blob, BnInOp2Blob(GenDiffBn("in")));
}

template<DeviceType device_type, typename T>
void InstanceStackKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  is_first_in_ = PieceSliceKernelUtil<device_type, T>::StackInstanceShape(
      is_first_in_, BnInOp2Blob("in"), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void InstanceStackKernel<device_type, T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK(in_blob->has_dim0_valid_num_field());
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_EQ(total_ins_num, out_blob->static_shape().At(0));
  out_blob->set_dim1_valid_num(ins_idx, in_blob->shape().At(0));
}

template<DeviceType device_type, typename T>
void InstanceStackKernel<device_type, T>::ForwardDim2ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK(in_blob->has_dim1_valid_num_field());
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_EQ(total_ins_num, out_blob->static_shape().At(0));
  FOR_RANGE(size_t, i, 0, in_blob->shape().At(0)) {
    out_blob->set_dim2_valid_num(ins_idx, i, in_blob->dim1_valid_num(i));
  }
}

template<DeviceType device_type, typename T>
void InstanceStackKernel<device_type, T>::BackwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  PieceSliceKernelUtil<device_type, T>::SliceInstanceShape(BnInOp2Blob(GenDiffBn("out")),
                                                           BnInOp2Blob(GenDiffBn("in")));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kInstanceStackConf, InstanceStackKernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
