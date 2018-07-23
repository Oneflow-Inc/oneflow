#ifndef ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class ProposalKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ProposalKernel);
  ProposalKernel() = default;
  ~ProposalKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T>
class ProposalKernelUtil final {
 public:
  static void BboxTransform(DeviceCtx* ctx, int64_t m, const T* bbox, const T* target_bbox,
                            T* deltas);
  static void BboxTransformInv(DeviceCtx* ctx, int64_t m, const T* bbox, const T* deltas,
                               T* bbox_pred);
  static void TakeFgScores(DeviceCtx* ctx, int64_t m, int64_t a, const T* class_prob, T* fg_scores);
  static void ClipBoxes(DeviceCtx* ctx, int64_t n, int64_t m, const float* im_info, T* proposals);
  static std::vector<int64_t> FilterBoxesByMinSize(DeviceCtx* ctx, int64_t n, int64_t m,
                                                   int32_t min_size, const float* im_info,
                                                   T* proposals);
  static void SortByScore(DeviceCtx* ctx, int64_t n, int64_t m, std::vector<int64_t> keep_to,
                          T* fg_score, T* proposals);
  static void Nms(DeviceCtx* ctx, int64_t n, int64_t m, const std::vector<int64_t>& keep_to,
                  int64_t pre_nms_top_n, int64_t post_nms_top_n, float nms_threshold,
                  const T* proposals, const T* fg_scores, T* rois, T* scores);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_PROPOSAL_KERNEL_H_
