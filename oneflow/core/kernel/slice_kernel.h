#ifndef ONEFLOW_CORE_KERNEL_SLICE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SLICE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SliceKernel;

template<typename T>
class SliceKernel<DeviceType::kCPU, T> final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceKernel);
  SliceKernel() = default;
  ~SliceKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<typename T>
class SliceKernel<DeviceType::kGPU, T> final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceKernel);
  SliceKernel() = default;
  ~SliceKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void InitOut2InOffsetFromHost(DeviceCtx* ctx, const Shape& in_shape, Blob* blob) const;
};

template<typename T, size_t NDIMS>
struct NdarraySliceUtil;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SLICE_KERNEL_H_
