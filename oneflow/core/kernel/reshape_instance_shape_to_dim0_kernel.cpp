#include "oneflow/core/kernel/reshape_instance_shape_to_dim0_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReshapeInstanceShapeToDim0Kernel<device_type>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const Shape out_instance_shape(this->op_conf().reshape_instance_shape_to_dim0_conf().shape());
  CHECK_EQ(in->shape().elem_cnt() % out_instance_shape.elem_cnt(), 0);
  const int64_t out_dim0_valid_num = in->shape().elem_cnt() / out_instance_shape.elem_cnt();
  out->set_dim0_valid_num(0, out_dim0_valid_num);
}

template<DeviceType device_type>
void ReshapeInstanceShapeToDim0Kernel<device_type>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

template<DeviceType device_type>
void ReshapeInstanceShapeToDim0Kernel<device_type>::BackwardInDiffDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

template<DeviceType device_type>
void ReshapeInstanceShapeToDim0Kernel<device_type>::BackwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  in_diff->set_instance_shape(in->instance_shape());
  if (in->has_dim0_valid_num_field()) { in_diff->CopyDim0ValidNumFrom(ctx.device_ctx, in); }
}

template<DeviceType device_type>
void ReshapeInstanceShapeToDim0Kernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt());
  out->CopyDataContentFrom(ctx.device_ctx, in);
}

template<DeviceType device_type>
void ReshapeInstanceShapeToDim0Kernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  CHECK_EQ(in_diff->shape().elem_cnt(), out_diff->shape().elem_cnt());
  in_diff->CopyDataContentFrom(ctx.device_ctx, out_diff);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReshapeInstanceShapeToDim0Conf,
                               ReshapeInstanceShapeToDim0Kernel);

}  // namespace oneflow
