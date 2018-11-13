#ifndef ONEFLOW_CORE_KERNEL_SLICE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SLICE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SliceKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceKernel);
  SliceKernel() = default;
  ~SliceKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct SliceKernelUtil final {
  static void Forward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                      const Blob* in_blob, Blob* out_blob);
  static void Backward(DeviceCtx* device_ctx, const PbRpf<DimSliceConf>& rep_dim_slice,
                       const Blob* out_diff_blob, Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SLICE_KERNEL_H_
