#include "oneflow/core/kernel/proposal_target_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"

namespace oneflow {

namespace {

template<typename T>
void RoisNearestGtAndMaxIou(const int64_t rois_num, const int64_t gt_num, const T* rpn_rois_ptr,
                            const T* gt_boxes_ptr, int32_t* roi_nearest_gt_index_ptr,
                            T* roi_max_overlap_ptr) {
  const BBox<T>* roi_box = BBox<T>::Cast(rpn_rois_ptr);
  const BBox<T>* gt_box = BBox<T>::Cast(gt_boxes_ptr);
  FOR_RANGE(int64_t, i, 0, rois_num) {
    float maxIou = 0;
    int64_t maxIouIdx = 0;
    FOR_RANGE(int64_t, j, 0, gt_num) {
      float iou = roi_box[j].InterOverUnion(&gt_box[j]);
      if (iou > maxIou) {
        maxIou = iou;
        maxIouIdx = j;
      }
    }
    roi_nearest_gt_index_ptr[i] = maxIouIdx;
    roi_max_overlap_ptr[i] = maxIou;
  }
}

template<typename T>
void SortRoisIndexByIou(const int64_t rois_num, const T* roi_max_overlap_ptr,
                        int32_t* rois_index_ptr) {
  FOR_RANGE(int64_t, i, 0, rois_num) { rois_index_ptr[i] = i; }
  std::sort(rois_index_ptr, rois_index_ptr + rois_num, [&](int32_t lhs, int32_t rhs) {
    return roi_max_overlap_ptr[lhs] > roi_max_overlap_ptr[rhs];
  });
}
void SampleChoice(const int64_t startIdx, const int64_t endIdx, int32_t* rois_index_ptr,
                  const int32_t sample_size) {
  if (endIdx - startIdx == sample_size) { return; }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(rois_index_ptr + startIdx, rois_index_ptr + endIdx, gen);
}
template<typename T>
void FilterAndChoiceFgBg(const ProposalTargetOpConf& conf, const int64_t rois_num,
                         const T* roi_max_overlap_ptr, int32_t* rois_index_ptr,
                         int64_t& fg_start_idx, int64_t& bg_start_idx) {
  // const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const int64_t fg_thresh = conf.fg_thresh();
  const int64_t bg_thresh_hi = conf.bg_thresh_hi();
  const int64_t bg_thresh_lo = conf.bg_thresh_lo();
  CHECK_GT(bg_thresh_hi, bg_thresh_lo);
  int64_t fg_end_index = 0;
  int64_t bg_start_index = 0;
  int64_t bg_end_index = 0;
  // include start not include end
  FOR_RANGE(int64_t, i, 0, rois_num) {
    int64_t rois_idx = rois_index_ptr[i];
    if (roi_max_overlap_ptr[rois_idx] >= fg_thresh) {
      if ((i + 1 < rois_num && roi_max_overlap_ptr[rois_index_ptr[i + 1]] < fg_thresh)
          || (i + 1 == rois_num)) {
        fg_end_index = i + 1;
      }
    }
    if (roi_max_overlap_ptr[rois_idx] >= bg_thresh_hi) {
      if ((i + 1 < rois_num && roi_max_overlap_ptr[rois_index_ptr[i + 1]] < bg_thresh_hi)
          || (i + 1 == rois_num)) {
        bg_start_index = i + 1;
      }
    }
    if (roi_max_overlap_ptr[rois_idx] >= bg_thresh_lo) {
      if ((i + 1 < rois_num && roi_max_overlap_ptr[rois_index_ptr[i + 1]] < bg_thresh_lo)
          || (i + 1 == rois_num)) {
        bg_end_index = i + 1;
      }
    }
  }
  const int64_t num_roi_per_image = conf.num_roi_per_image();
  const float fg_fraction = conf.fg_fraction();
  const int64_t fg_sample_size =
      std::min(fg_end_index + 1, (int64_t)std::round(num_roi_per_image * fg_fraction));
  SampleChoice(
      0, fg_end_index, rois_index_ptr,
      fg_sample_size);  // fg_sample: rois_index_ptr(fg_start_index,fg_start_index+fg_sample_size)
  const int64_t bg_sample_size =
      std::min(bg_end_index - bg_start_index + 1, num_roi_per_image - fg_sample_size);
  SampleChoice(
      bg_start_index, bg_end_index, rois_index_ptr,
      bg_sample_size);  // bg_sample: rois_index_ptr(bg_start_index,bg_start_index+bg_sample_size)
}

template<typename T>
void ComputeTargetAndWriteOut(const int32_t* roi_index_ptr, const int32_t* roi_nearest_gt_index_ptr,
                              const T* rpn_rois_ptr, const T* gt_boxes_ptr,
                              const int32_t* gt_labels_ptr, T* rois_ptr, int32_t* labels_ptr,
                              T* bbox_targets_ptr, T* inside_weights_ptr, T* outside_weights_ptr) {
  TODO();
}

}  // namespace

template<typename T>
void ProposalTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* rpn_rois_blob = BnInOp2Blob("rpn_rois");               //(im_num, roi, 4)
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");               //(im_num,gt_max_num,4)
  const Blob* gt_label_blob = BnInOp2Blob("gt_label");               //(im_num,gt_max_num,1)
  const Blob* im_info_blob = BnInOp2Blob("im_info");                 //(im_num,3)
  Blob* rois_blob = BnInOp2Blob("rois");                             //(im_num, sample_num, 4)
  Blob* labels_blob = BnInOp2Blob("labels");                         //(im_num, sample_num, 1)
  Blob* bbox_targets_blob = BnInOp2Blob("bbox_targets");             //(im_num, sample_num, 4*class)
  Blob* inside_weights_blob = BnInOp2Blob("bbox_inside_weights");    //(im_num, sample_num, 4*class)
  Blob* outside_weights_blob = BnInOp2Blob("bbox_outside_weights");  //(im_num, sample_num, 4*class)
  // tmp blob
  int32_t* roi_nearest_gt_index_ptr =
      BnInOp2Blob("roi_nearest_gt_index")->mut_dptr<int32_t>();              //(roi)
  T* roi_max_overlap_ptr = BnInOp2Blob("roi_max_overlap")->mut_dptr<T>();    //(roi)
  int32_t* rois_index_ptr = BnInOp2Blob("rois_index")->mut_dptr<int32_t>();  //(roi)
  int64_t im_num = rpn_rois_blob->shape().At(0);
  int64_t roi_num = rpn_rois_blob->shape().At(1);
  // int64_t gt_max_num = gt_boxes_blob->shape().At(1);
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  FOR_RANGE(int64_t, i, 0, im_num) {
    const T* rpn_rois_ptr = rois_blob->dptr<T>(i);
    const T* gt_boxes_ptr = gt_boxes_blob->dptr<T>(i);
    const int32_t* gt_labels_ptr = gt_label_blob->dptr<int32_t>(i);
    int32_t gt_num = im_info_blob->dptr<int32_t>(i)[2];
    T* rois_ptr = rois_blob->mut_dptr<T>(i);
    int32_t* labels_ptr = labels_blob->mut_dptr<int32_t>(i);
    T* bbox_targets_ptr = bbox_targets_blob->mut_dptr<T>(i);
    T* inside_weights_ptr = inside_weights_blob->mut_dptr<T>(i);
    T* outside_weights_ptr = outside_weights_blob->mut_dptr<T>(i);
    RoisNearestGtAndMaxIou(roi_num, gt_num, rpn_rois_ptr, gt_boxes_ptr, roi_nearest_gt_index_ptr,
                           roi_max_overlap_ptr);
    SortRoisIndexByIou(roi_num, roi_max_overlap_ptr, rois_index_ptr);
    int64_t fg_start_index = 0;
    int64_t bg_start_index = 0;
    FilterAndChoiceFgBg(conf, roi_num, roi_max_overlap_ptr, rois_index_ptr, fg_start_index,
                        bg_start_index);
    ComputeTargetAndWriteOut(rois_index_ptr, roi_nearest_gt_index_ptr, rpn_rois_ptr, gt_boxes_ptr,
                             gt_labels_ptr, rois_ptr, labels_ptr, bbox_targets_ptr,
                             inside_weights_ptr, outside_weights_ptr);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalTargetConf, ProposalTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
