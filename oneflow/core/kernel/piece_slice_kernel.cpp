#include "oneflow/core/kernel/piece_slice_kernel.h"
#include "oneflow/core/kernel/piece_slice_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void PieceSliceKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_EQ(total_ins_num, in_blob->static_shape().At(0));
  PieceSliceKernelUtil<device_type>::PieceSlice(ctx.device_ctx, ins_idx, total_ins_num, in_blob,
                                                BnInOp2Blob("out"));
}

template<DeviceType device_type>
void PieceSliceKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t out_diff_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(out_diff_idx, total_ins_num);
  CHECK_EQ(total_ins_num, in_diff_blob->static_shape().At(0));
  PieceSliceKernelUtil<device_type>::InstanceStack(ctx.device_ctx, out_diff_idx, total_ins_num,
                                                   BnInOp2Blob(GenDiffBn("out")), in_diff_blob);
}

template<DeviceType device_type>
void PieceSliceKernel<device_type>::ForwardDim0ValidNum(
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
    out_blob->set_dim0_valid_num(0, in_blob->shape().At(0));
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type>
void PieceSliceKernel<device_type>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const bool uncontiguous_varing_instance =
      in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field();
  const bool contiguous_varing_instance = in_blob->has_instance_shape_field();
  if (contiguous_varing_instance) {
    CHECK(!uncontiguous_varing_instance);
    BnInOp2Blob("out")->set_instance_shape(
        Shape(std::vector<int64_t>(in_blob->instance_shape().dim_vec().begin() + 1,
                                   in_blob->instance_shape().dim_vec().end())));
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type>
void PieceSliceKernel<device_type>::ForwardDim1ValidNum(
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
    FOR_RANGE(size_t, i, 0, out_blob->shape().At(0)) {
      out_blob->set_dim1_valid_num(i, in_blob->dim2_valid_num(ins_idx, i));
    }
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type>
void PieceSliceKernel<device_type>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kPieceSliceConf, PieceSliceKernel);

}  // namespace oneflow
