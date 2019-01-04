#include "oneflow/core/kernel/resize_nearest_neighbor_kernel.h"
#include "oneflow/core/operator/operator_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ResizeNearestNeighborKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const Shape& in_shape = in_blob->shape();
  const Shape& out_shape = out_blob->shape();
  const bool align_corners = this->op_conf().resize_nearest_neighbor_conf().align_corners();
  ResizeNearestNeighborUtil<device_type, T>::Forward(
      ctx, GetResizeScale(in_shape.At(2), out_shape.At(2), align_corners),
      GetResizeScale(in_shape.At(3), out_shape.At(3), align_corners), align_corners, in_blob,
      out_blob);
}

template<DeviceType device_type, typename T>
void ResizeNearestNeighborKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Shape& in_shape = in_diff_blob->shape();
  const Shape& out_shape = out_diff_blob->shape();
  const bool align_corners = this->op_conf().resize_nearest_neighbor_conf().align_corners();
  ResizeNearestNeighborUtil<device_type, T>::Backward(
      ctx, GetResizeScale(in_shape.At(2), out_shape.At(2), align_corners),
      GetResizeScale(in_shape.At(3), out_shape.At(3), align_corners), align_corners, out_diff_blob,
      in_diff_blob);
}

template<typename T>
struct ResizeNearestNeighborUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const float scale_h, const float scale_w,
                      const bool align_corners, const Blob* in_blob, Blob* out_blob) {
    UNIMPLEMENTED();
  }
  static void Backward(const KernelCtx& ctx, const float scale_h, const float scale_w,
                       const bool align_corners, const Blob* out_diff_blob, Blob* in_diff_blob) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kResizeNearestNeighborConf, ResizeNearestNeighborKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
