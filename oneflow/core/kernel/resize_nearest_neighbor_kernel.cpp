#include "oneflow/core/kernel/resize_nearest_neighbor_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ResizeNearestNeighbor<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  const ResizeNearestNeighborKernelConf& kernel_conf =
      this->kernel_conf().resize_nearest_neighbor_conf();
  ResizeNearestNeighborUtil<device_type, T>::Forward(
      ctx, kernel_conf, this->op_conf().resize_nearest_neighbor_conf().align_corners(), in_blob,
      out_blob);
}

template<DeviceType device_type, typename T>
void ResizeNearestNeighbor<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const ResizeNearestNeighborKernelConf& kernel_conf =
      this->kernel_conf().resize_nearest_neighbor_conf();
  ResizeNearestNeighborUtil<device_type, T>::Backward(
      ctx, kernel_conf, this->op_conf().resize_nearest_neighbor_conf().align_corners(),
      out_diff_blob, in_diff_blob);
}

template<typename T>
struct ResizeNearestNeighborUtil<DeviceType::kCPU, T> {
  static void Forward(const KernelCtx& ctx, const ResizeNearestNeighborKernelConf& kernel_conf,
                      const bool align_corners, const Blob* in_blob, Blob* out_blob) {
    UNIMPLEMENTED();
  }
  static void Backward(const KernelCtx& ctx, const ResizeNearestNeighborKernelConf& kernel_conf,
                       const bool align_corners, const Blob* out_diff_blob, Blob* in_diff_blob) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kResizeNearestNeighborConf, ResizeNearestNeighbor,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kUpsampleNearestConf, ResizeNearestNeighbor,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
