#include "oneflow/core/kernel/resize_nearest_neighbor_kernel.h"
#include "oneflow/core/kernel/upsample_nearest_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void UpsampleNearestKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const int32_t scale = this->op_conf().upsample_nearest_conf().scale();
  ResizeNearestNeighborUtil<device_type, T>::Forward(
      ctx, 1.f/scale, 1.f/scale,false, in_blob,
      out_blob);
}

template<DeviceType device_type, typename T>
void UpsampleNearestKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const int32_t scale = this->op_conf().upsample_nearest_conf().scale();
  UpsampleNearestUtil<device_type, T>::Backward(
      ctx, 1.f/scale, 1.f/scale, false, out_diff_blob, in_diff_blob);
}

template<DeviceType device_type, typename T>
void UpsampleNearestKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int32_t scale = this->op_conf().upsample_nearest_conf().scale();
  const Shape& in_shape = BnInOp2Blob("in")->shape();
  CHECK_EQ(in_shape.NumAxes(), 4);
  BnInOp2Blob("out")->set_instance_shape(Shape({in_shape.At(1), scale * in_shape.At(2), scale * in_shape.At(3)}));
}

template<DeviceType device_type, typename T>
void UpsampleNearestKernel<device_type, T>::BackwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (!in_diff_blob) { return; }
  const int32_t scale = this->op_conf().upsample_nearest_conf().scale();
  const Shape& out_shape = BnInOp2Blob("out_diff")->shape();
  CHECK_EQ(out_shape.At(2) % scale, 0)
  CHECK_EQ(out_shape.At(3) % scale, 0)
  in_diff_blob->set_instance_shape(Shape({out_shape.At(1), out_shape.At(2) / scale, out_shape.At(3) / scale});
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kUpsampleNearestConf, UpsampleNearestKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
