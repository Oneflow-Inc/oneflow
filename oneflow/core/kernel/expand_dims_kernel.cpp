#include "oneflow/core/kernel/expand_dims_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ExpandDimsKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

template<DeviceType device_type>
void ExpandDimsKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
}

template<DeviceType device_type>
void ExpandDimsKernel<device_type>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int32_t instance_shape_axis = this->op_conf().expand_dims_conf().axis() - 1;
  std::vector<int64_t> dim_vec = BnInOp2Blob("in")->instance_shape().dim_vec();
  CHECK_GE(instance_shape_axis, 0);
  CHECK_LE(instance_shape_axis, dim_vec.size());
  dim_vec.insert(dim_vec.begin() + instance_shape_axis, 1);
  BnInOp2Blob("out")->set_instance_shape(Shape(dim_vec));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kExpandDimsConf, ExpandDimsKernel);

}  // namespace oneflow
