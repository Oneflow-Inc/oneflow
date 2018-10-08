#ifndef ONEFLOW_CORE_KERNEL_RESIZE_NEAREST_NEIGHBOR_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RESIZE_NEAREST_NEIGHBOR_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ResizeNearestNeighbor final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ResizeNearestNeighbor);
  ResizeNearestNeighbor() = default;
  ~ResizeNearestNeighbor() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct ResizeNearestNeighborUtil {
  static void Forward(const KernelCtx& ctx, const ResizeNearestNeighborKernelConf& kernel_conf,
                      const bool align_corners, const Blob* in_blob, Blob* out_blob);
  static void Backward(const KernelCtx& ctx, const ResizeNearestNeighborKernelConf& kernel_conf,
                       const bool align_corners, const Blob* out_diff_blob, Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RESIZE_NEAREST_NEIGHBOR_KERNEL_H_
