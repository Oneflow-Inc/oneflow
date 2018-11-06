#ifndef ONEFLOW_CORE_KERNEL_FPN_COLLECT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_FPN_COLLECT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class FpnCollectKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FpnCollectKernel);
  FpnCollectKernel() = default;
  ~FpnCollectKernel() = default;

  using BBox = IndexedBBoxT<const T>;
  using MutBBox = IndexedBBoxT<T>;

 private:
  void VirtualKernelInit(const ParallelContext* parallel_ctx) override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void ForwardDim0ValidNum(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardRecordIdInDevicePiece(
      const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::vector<std::vector<int32_t>> GroupRoiBoxes(const std::vector<const Blob*>& rois_fpn_blobs,
                                                  const int32_t row_len) const;
  std::vector<size_t> GetRoiGroupSizeVec(
      const std::vector<std::vector<int32_t>>& im_grouped_roi_inds_vec) const;

  int32_t num_groups_;
  bool need_group_by_img_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_
