#ifndef ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class AnchorTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AnchorTargetKernel);
  AnchorTargetKernel() = default;
  ~AnchorTargetKernel() = default;

  using BBox = BBoxT<const T>;
  using MutBBox = BBoxT<T>;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  void ForwardInstanceShape(const KernelCtx& ctx,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ClearOutputBlobs(DeviceCtx* ctx,
                        const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void GenerateAnchorBoxes(DeviceCtx* ctx,
                           const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void FilterOutsideAnchorBoxes(DeviceCtx* ctx, int64_t im_index,
                                const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void CalcMaxOverlapAndSetPositiveLabels(
      DeviceCtx* ctx, int64_t im_index,
      const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  size_t Subsample(DeviceCtx* ctx, int64_t im_index,
                   const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void OutputForEachImage(DeviceCtx* ctx, int64_t im_index,
                          const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
