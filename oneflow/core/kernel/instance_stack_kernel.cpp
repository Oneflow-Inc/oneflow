#include "oneflow/core/kernel/instance_stack_kernel.h"
#include "oneflow/core/kernel/piece_slice_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void InstanceStackKernel<device_type>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  is_first_in_ = true;
}

template<DeviceType device_type>
void InstanceStackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t ins_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(ins_idx, total_ins_num);
  CHECK_EQ(total_ins_num, out_blob->static_shape().At(0));
  PieceSliceKernelUtil<device_type>::InstanceStack(ctx.device_ctx, ins_idx, total_ins_num,
                                                   BnInOp2Blob("in"), out_blob);
}

template<DeviceType device_type>
void InstanceStackKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  auto* actor_stat = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  const size_t in_diff_idx = actor_stat->first;
  const size_t total_ins_num = actor_stat->second;
  CHECK_LE(in_diff_idx, total_ins_num);
  CHECK_EQ(total_ins_num, out_diff_blob->static_shape().At(0));
  PieceSliceKernelUtil<device_type>::PieceSlice(ctx.device_ctx, in_diff_idx, total_ins_num,
                                                out_diff_blob, BnInOp2Blob(GenDiffBn("in")));
}

template<DeviceType device_type>
void InstanceStackKernel<device_type>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK(in_blob->has_instance_shape_field());
  CHECK(!(in_blob->has_dim1_valid_num_field() || in_blob->has_dim2_valid_num_field()));
  CHECK(out_blob->has_instance_shape_field());
  if (is_first_in_) {
    BnInOp2Blob("out")->set_instance_shape(in_blob->shape());
  } else {
    CHECK_EQ(in_blob->shape(), Shape(std::vector<int64_t>(out_blob->shape().dim_vec().begin() + 1,
                                                          out_blob->shape().dim_vec().end())));
  }
  is_first_in_ = false;
}

template<DeviceType device_type>
void InstanceStackKernel<device_type>::ForwardDim1ValidNum(
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

template<DeviceType device_type>
void InstanceStackKernel<device_type>::ForwardDim2ValidNum(
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

template<DeviceType device_type>
void InstanceStackKernel<device_type>::BackwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  if (in_diff_blob->shape().NumAxes() < 2) { return; }
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  const bool uncontiguous_varing_instance =
      out_diff_blob->has_dim1_valid_num_field() || out_diff_blob->has_dim2_valid_num_field();
  const bool contiguous_varing_instance = out_diff_blob->has_instance_shape_field();
  if (contiguous_varing_instance) {
    CHECK(!uncontiguous_varing_instance);
    in_diff_blob->set_instance_shape(
        Shape(std::vector<int64_t>(out_diff_blob->instance_shape().dim_vec().begin() + 1,
                                   out_diff_blob->instance_shape().dim_vec().end())));
  } else {
    UNIMPLEMENTED();
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kInstanceStackConf, InstanceStackKernel);

}  // namespace oneflow
