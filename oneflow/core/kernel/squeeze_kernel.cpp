#include "oneflow/core/kernel/squeeze_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void SqueezeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
}

template<DeviceType device_type>
void SqueezeKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  in_diff_blob->CopyDataContentFrom(ctx.device_ctx, out_diff_blob);
}

template<DeviceType device_type>
void SqueezeKernel<device_type>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  auto conf = this->op_conf().squeeze_conf();
  CHECK_EQ(in_blob->instance_shape().NumAxes() - conf.axis_size(),
           out_blob->instance_shape().NumAxes());

  const std::vector<int32_t> vec = PbRf2StdVec(conf.axis());
  int32_t idx = 0;
  FOR_RANGE(int32_t, i, 0, in_blob->instance_shape().NumAxes()) {
    if (std::find(vec.begin(), vec.end(), i) == vec.end()) {
      out_blob->mut_instance_shape_ptr()[idx++] = in_blob->instance_shape().At(i);
    }
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kSqueezeConf, SqueezeKernel);

}  // namespace oneflow
