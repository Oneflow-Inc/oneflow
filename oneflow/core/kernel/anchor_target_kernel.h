#ifndef ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

template<typename T>
class AnchorTargetKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AnchorTargetKernel);
  AnchorTargetKernel() = default;
  ~AnchorTargetKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

class AnchorLabelsAndMaxOverlapsInfo final {
public:
  AnchorLabelsAndMaxOverlapInfo(int32_t* anchor_labels_ptr, int32_t* max_overlaps_ptr, 
                                float* max_overlap_gt_boxes_idx_ptr, 
                                float positive_threshold, float negative_threshold,
                                size_t size, bool init_label = true)
      : anchor_labels_ptr_(anchor_labels_ptr)
      , max_overlaps_ptr_(max_overlaps_ptr)
      , max_overlap_gt_boxes_idx_ptr_(max_overlap_gt_boxes_idx_ptr)
      , positive_threshold_(positive_threshold)
      , negative_threshold_(negative_threshold)
      , size_(size) {
    if (init_label) {
      std::fill(anchor_labels_ptr_, anchor_labels_ptr_ + size_, -1);
    }
  }
  
  void AssignLabelByOverlapThreshold(int32_t anchor_idx, int32_t gt_box_idx, float overlap) {
    CHECK_LT(anchor_idx, size_);
    int32_t cand_label = -1;
    if (overlap >= positive_threshold_) {
      cand_label = 1;
    } else if (overlap < negative_threshold_) {
      cand_label = 0;
    }
    if (cand_label >= anchor_labels_ptr_[anchor_idx]) {
      anchor_labels_ptr_[anchor_idx] = cand_label;
      if (overlap >= max_overlaps_ptr_[anchor_idx]) {
        max_overlaps_ptr_[anchor_idx] = overlap;
        max_overlap_gt_boxes_idx_ptr_[anchor_idx] = gt_box_idx;
      }
    }
  }

  void TrySetPositiveLabel(int32_t anchor_idx) {
    CHECK_LT(anchor_idx, size_);
    if (anchor_labels_ptr_[anchor_idx] != 0) {
      anchor_labels_ptr_[anchor_idx] == 1;
    }
  }

private:
  int32_t* anchor_labels_ptr_;
  float* max_overlaps_ptr_;
  int32_t* max_overlap_gt_boxes_idx_ptr_;
  const float positive_threshold_;
  const float negative_threshold_;
  const size_t size_;
};

class GtBoxesNearestAnchorsInfo final {
public:
  GtBoxesNearestAnchorsInfo(int32_t* anchors_idx_ptr, float* overlap_ptr)
      : gt_max_overlaps_ptr_(overlap_ptr)
      , nearest_anchors_idx_ptr_(anchors_idx_ptr)
      , last_gt_box_idx_(-1)
      , last_gt_box_record_end_(0)
      , record_anchors_num_(0) { }

  void TryRecordAnchorAsNearest(int32_t gt_box_idx, int32_t anchor_idx, float overlap) {
    if (gt_box_idx != last_gt_box_idx_) {
      last_gt_box_record_end_ = record_anchors_num_;
    }
    if (overlap >= gt_max_overlaps_ptr_[gt_box_idx]) {
      if (overlap > gt_max_overlaps_ptr_[gt_box_idx]) {
        record_anchors_num_ = last_gt_box_record_end_;
      }
      nearest_anchors_idx_ptr_[++record_anchors_num_] = anchor_idx;
    }
  }

  void ForEachNearestAnchor(const std::function<void(int32_t)>& Handler) {
    FOR_RANGE(int32_t, i, 0, record_anchors_num_) {
      Handler(nearest_anchors_idx_ptr_[i]);
    }
  }

private:
  float* gt_max_overlaps_ptr_;
  int32_t* nearest_anchors_idx_ptr_;
  int32_t last_gt_box_idx_;
  int32_t last_gt_box_record_end_;
  int32_t record_anchors_num_;
}

}  // namespace oneflow
#endif  // ONEFLOW_CORE_OPERATOR_ANCHOR_TARGET_KERNEL_OP_H_
