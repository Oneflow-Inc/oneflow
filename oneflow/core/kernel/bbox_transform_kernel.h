#ifndef ONEFLOW_CORE_KERNEL_BBOX_TRANSFORM_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BBOX_TRANSFORM_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BboxTransformKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BboxTransformKernel);
  BboxTransformKernel() = default;
  ~BboxTransformKernel() = default;

  using BBox = BBoxImpl<T, BBoxBase, BBoxCoord::kCorner>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDim1ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_TRANSFORM_KERNEL_H_