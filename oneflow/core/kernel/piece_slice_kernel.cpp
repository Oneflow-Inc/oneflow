#include "oneflow/core/kernel/piece_slice_kernel.h"
#include "oneflow/core/kernel/piece_slice_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void PieceSliceKernel<device_type, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  is_first_out_diff_ = true;
}

template<DeviceType device_type, typename T>
void PieceSliceKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_EQ(total_ins_num, in_blob->static_shape().At(0));
  PieceSliceKernelUtil<device_type, T>::PieceSlice(ctx.device_ctx, ins_idx, total_ins_num, in_blob,
                                                   BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void PieceSliceKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t out_diff_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(out_diff_idx, total_ins_num);
  CHECK_EQ(total_ins_num, in_diff_blob->static_shape().At(0));
  PieceSliceKernelUtil<device_type, T>::InstanceStack(ctx.device_ctx, out_diff_idx, total_ins_num,
                                                      BnInOp2Blob(GenDiffBn("out")), in_diff_blob);
}

template<DeviceType device_type, typename T>
void PieceSliceKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_EQ(total_ins_num, in_blob->static_shape().At(0));
  const bool uncontiguous_varing_instance =
      in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field();
  const bool contiguous_varing_instance = in_blob->has_instance_shape_field();
  if (uncontiguous_varing_instance) {
    CHECK(!contiguous_varing_instance);
    CHECK(in_blob->has_dim1_valid_num_field());
    out_blob->set_dim0_valid_num(0, in_blob->dim1_valid_num(ins_idx));
  } else if (contiguous_varing_instance) {
    CHECK(!uncontiguous_varing_instance);
    out_blob->set_dim0_valid_num(0, in_blob->shape().At(1));
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void PieceSliceKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  PieceSliceKernelUtil<device_type, T>::SliceInstanceShape(BnInOp2Blob("in"), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void PieceSliceKernel<device_type, T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_EQ(total_ins_num, in_blob->static_shape().At(0));
  const bool uncontiguous_varing_instance =
      in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field();
  const bool contiguous_varing_instance = in_blob->has_instance_shape_field();
  if (uncontiguous_varing_instance) {
    CHECK(!contiguous_varing_instance);
    FOR_RANGE(size_t, i, 0, in_blob->dim1_valid_num(ins_idx)) {
      out_blob->set_dim1_valid_num(i, in_blob->dim2_valid_num(ins_idx, i));
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void PieceSliceKernel<device_type, T>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<DeviceType device_type, typename T>
void PieceSliceKernel<device_type, T>::BackwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  is_first_out_diff_ = PieceSliceKernelUtil<device_type, T>::StackInstanceShape(
      is_first_out_diff_, BnInOp2Blob(GenDiffBn("out")), BnInOp2Blob(GenDiffBn("in")));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kPieceSliceConf, PieceSliceKernel,
                           ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
