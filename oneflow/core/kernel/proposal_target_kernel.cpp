#include "oneflow/core/kernel/proposal_kernel.h"
#include "oneflow/core/kernel/proposal_target_kernel.h"
#include "oneflow/core/kernel/rcnn_util.h"

namespace oneflow {
namespace {
template<typename T>
void IncludeGt2Rois(const T* rois_dptr, int32_t rois_num, const T* gt_box_dptr, int32_t gt_num,
                    T* all_rois_dptr) {
  int n = rois_num * 4;
  FOR_RANGE(int32_t, i, 0, n) { all_rois_dptr[i] = rois_dptr[i]; }
  FOR_RANGE(int32_t, i, 0, gt_num) {
    int32_t all_rois_idx = (rois_num + i) * 4;
    all_rois_dptr[all_rois_idx] = gt_box_dptr[i * 5];
    all_rois_dptr[all_rois_idx + 1] = gt_box_dptr[i * 5 + 1];
    all_rois_dptr[all_rois_idx + 2] = gt_box_dptr[i * 5 + 2];
    all_rois_dptr[all_rois_idx + 3] = gt_box_dptr[i * 5 + 3];
  }
}
}  // namespace
template<typename T>
void ProposalTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* rpn_rois_blob = BnInOp2Blob("rpn_rois");
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  const Blob* gt_num_blob = BnInOp2Blob("gt_num");
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* labels_blob = BnInOp2Blob("labels");
  Blob* bbox_targets_blob = BnInOp2Blob("bbox_targets");
  Blob* inside_weights_blob = BnInOp2Blob("bbox_inside_weights");
  Blob* outside_weights_blob = BnInOp2Blob("bbox_outside_weights");
  // data_tmp blob
  Blob* all_rois_blob = BnInOp2Blob("all_rois");
  Blob* overlap_blob = BnInOp2Blob("bbox_overlap");
  Blob* labels_tmp_blob = BnInOp2Blob("labels_tmp");
  Blob* roi_argmax_blob = BnInOp2Blob("roi_argmax");
  Blob* max_overlaps_blob = BnInOp2Blob("max_overlaps");
  Blob* fg_inds_blob = BnInOp2Blob("fg_inds");  // fg_indexs(box_num,1)
  Blob* bg_inds_blob = BnInOp2Blob("bg_inds");  // bg_indexs(box_num,1)
  Blob* fg_bg_sample_inds_blob = BnInOp2Blob("fg_bg_sample_inds");
  const int32_t num_roi_per_image = this->op_conf().proposal_target_conf().num_roi_per_image();
  const float fg_fraction = this->op_conf().proposal_target_conf().fg_fraction();
  const int32_t fg_thresh = this->op_conf().proposal_target_conf().fg_thresh();
  const int32_t bg_thresh_hi = this->op_conf().proposal_target_conf().bg_thresh_hi();
  const int32_t bg_thresh_lo = this->op_conf().proposal_target_conf().bg_thresh_lo();
  const int32_t weight_x =
      this->op_conf().proposal_target_conf().bbox_inside_weight_conf().weight_x();
  const int32_t weight_y =
      this->op_conf().proposal_target_conf().bbox_inside_weight_conf().weight_y();
  const int32_t weight_w =
      this->op_conf().proposal_target_conf().bbox_inside_weight_conf().weight_w();
  const int32_t weight_h =
      this->op_conf().proposal_target_conf().bbox_inside_weight_conf().weight_h();
  const bool bbox_normalize_targets_precomputed =
      this->op_conf().proposal_target_conf().bbox_normalize_targets_precomputed();
  const int32_t means_x = this->op_conf().proposal_target_conf().bbox_normalize_means().means_x();
  const int32_t means_y = this->op_conf().proposal_target_conf().bbox_normalize_means().means_y();
  const int32_t means_w = this->op_conf().proposal_target_conf().bbox_normalize_means().means_w();
  const int32_t means_h = this->op_conf().proposal_target_conf().bbox_normalize_means().means_h();
  const int32_t stds_x = this->op_conf().proposal_target_conf().bbox_normalize_stds().stds_x();
  const int32_t stds_y = this->op_conf().proposal_target_conf().bbox_normalize_stds().stds_y();
  const int32_t stds_w = this->op_conf().proposal_target_conf().bbox_normalize_stds().stds_w();
  const int32_t stds_h = this->op_conf().proposal_target_conf().bbox_normalize_stds().stds_h();
  // 1. Include ground - truth boxes in rois
  // 2. overlaps: (rois x gt_boxes)
  // 3. get gt_assignment(argmax)&max_overlaps(max) for overlaps
  // 4. label=gt[gt_assignment,4]
  // 5. Select foreground RoIs indexs as maxoverlaps>FG_THRESH
  // 6. Sample foreground regions
  // 7. Select background RoIs indexs as [BG_THRESH_LO, BG_THRESH_HI)
  // 8. Compute number of background RoIs to take from this image
  // 9. Sample background regions without replacement
  int32_t num_batch = rpn_rois_blob->shape().At(0);
  int32_t rois_num = rpn_rois_blob->shape().At(1);
  int32_t gt_max_num = gt_boxes_blob->shape().At(1);
  const T* rpn_rois_dptr = rpn_rois_blob->dptr<T>();
  const T* gt_bbox_dptr = gt_boxes_blob->dptr<T>();
  const T* gt_num_dptr = gt_num_blob->dptr<T>();
  T* labels_dptr = labels_blob->mut_dptr<T>();
  T* rois_dptr = rois_blob->mut_dptr<T>();
  T* bbox_targets_dptr = bbox_targets_blob->mut_dptr<T>();
  T* inside_weights_dptr = inside_weights_blob->mut_dptr<T>();
  T* outside_weights_dptr = outside_weights_blob->mut_dptr<T>();
  // tmp blob
  T* all_rois_dptr = all_rois_blob->mut_dptr<T>();                    //(roi_num + gt_max_num, 4)
  T* overlap_dptr = overlap_blob->mut_dptr<T>();                      //(roi_num,gt_max_num)
  T* roi_argmax_dptr = roi_argmax_blob->mut_dptr<T>();                //(roi_num,1)
  T* max_overlaps_dptr = max_overlaps_blob->mut_dptr<T>();            //(roi_num,1)
  T* labels_tmp_dptr = labels_tmp_blob->mut_dptr<T>();                //(roi_num,1)
  T* fg_inds_dptr = fg_inds_blob->mut_dptr<T>();                      //(roi_num,1)
  T* bg_inds_dptr = bg_inds_blob->mut_dptr<T>();                      //(roi_num,1)
  T* fg_bg_sample_inds_dptr = fg_bg_sample_inds_blob->mut_dptr<T>();  //(num_roi_per_image,1)
  FOR_RANGE(int32_t, i, 0, num_batch) {
    int32_t gt_num = gt_num_dptr[i];
    rpn_rois_dptr = rpn_rois_blob->dptr<T>() + i * rois_num * 4;
    gt_bbox_dptr = gt_boxes_blob->dptr<T>() + i * gt_max_num * 5;
    labels_dptr = labels_blob->mut_dptr<T>() + i * num_roi_per_image;
    rois_dptr = rois_blob->mut_dptr<T>() + i * num_roi_per_image * 4;
    bbox_targets_dptr = bbox_targets_blob->mut_dptr<T>() + i * num_roi_per_image * 4;
    inside_weights_dptr = inside_weights_blob->mut_dptr<T>() + i * num_roi_per_image * 4;
    outside_weights_dptr = outside_weights_blob->mut_dptr<T>() + i * num_roi_per_image * 4;
    // tmp blob do not need to offset , its size not include im_num axis.
    IncludeGt2Rois(rpn_rois_dptr, rois_num, gt_bbox_dptr, gt_num, all_rois_dptr);
    RcnnUtil<T>::BboxOverlaps(all_rois_dptr, rois_num, gt_bbox_dptr, gt_num, gt_max_num,
                              overlap_dptr);
    RcnnUtil<T>::OverlapRowArgMax7Max(overlap_dptr, rois_num, gt_num, gt_max_num, roi_argmax_dptr,
                                      max_overlaps_dptr);
    int32_t fgidx = 0;
    int32_t bgidx = 0;
    FOR_RANGE(int32_t, j, 0, rois_num) {
      labels_tmp_dptr[j] = gt_bbox_dptr[static_cast<int32_t>(roi_argmax_dptr[j]) * 5 + 4];
      if (max_overlaps_dptr[j] > fg_thresh) { fg_inds_dptr[fgidx++] = j; }
      if (max_overlaps_dptr[j] >= bg_thresh_lo && max_overlaps_dptr[j] < bg_thresh_hi) {
        bg_inds_dptr[bgidx++] = j;
      }
    }
    int32_t fg_inds_num = fgidx;
    int32_t bg_inds_num = bgidx;
    int fg_rois_per_image = fg_inds_num;
    int bg_rois_per_image = bg_inds_num;
    if (fg_inds_num > num_roi_per_image * fg_fraction) {
      fg_rois_per_image = num_roi_per_image * fg_fraction;
    }
    RcnnUtil<T>::SampleChoice(fg_inds_dptr, fg_inds_num, fg_bg_sample_inds_dptr, fg_rois_per_image);
    if (bg_inds_num > num_roi_per_image - fg_rois_per_image) {
      bg_rois_per_image = num_roi_per_image - fg_rois_per_image;
    }
    RcnnUtil<T>::SampleChoice(bg_inds_dptr, bg_inds_num, fg_bg_sample_inds_dptr + fg_rois_per_image,
                              bg_rois_per_image);
    int out_weight_x = (weight_x > 0 ? weight_x : 0);
    int out_weight_y = (weight_y > 0 ? weight_y : 0);
    int out_weight_w = (weight_w > 0 ? weight_w : 0);
    int out_weight_h = (weight_h > 0 ? weight_h : 0);
    FOR_RANGE(int32_t, j, 0, num_roi_per_image) {
      if (j < fg_rois_per_image)  // foreground
      {
        int32_t index = fg_bg_sample_inds_dptr[j];
        labels_dptr[j] = labels_tmp_dptr[index];
        rois_dptr[j] = all_rois_dptr[index];
        int32_t gt_index = roi_argmax_dptr[index];
        bbox_targets_dptr[j * 4] = gt_bbox_dptr[gt_index * 5];
        bbox_targets_dptr[j * 4 + 1] = gt_bbox_dptr[gt_index * 5 + 1];
        bbox_targets_dptr[j * 4 + 2] = gt_bbox_dptr[gt_index * 5 + 2];
        bbox_targets_dptr[j * 4 + 3] = gt_bbox_dptr[gt_index * 5 + 3];
        inside_weights_dptr[j * 4] = weight_x;
        inside_weights_dptr[j * 4 + 1] = weight_y;
        inside_weights_dptr[j * 4 + 2] = weight_w;
        inside_weights_dptr[j * 4 + 3] = weight_h;
        outside_weights_dptr[j * 4] = out_weight_x;
        outside_weights_dptr[j * 4 + 1] = out_weight_y;
        outside_weights_dptr[j * 4 + 2] = out_weight_w;
        outside_weights_dptr[j * 4 + 3] = out_weight_h;
      } else {
        labels_dptr[j] = 0;
        FOR_RANGE(int32_t, k, 0, 4) {
          bbox_targets_dptr[j * 4 + k] = 0;
          inside_weights_dptr[j * 4 + k] = 0;
          outside_weights_dptr[j * 4 + k] = 0;
        }
      }
    }
  }
  ProposalKernelUtil<DeviceType::kCPU, T>::BboxTransform(ctx.device_ctx,
                                                           num_batch * num_roi_per_image, rois_dptr,
                                                            bbox_targets_dptr, bbox_targets_dptr);
  if (bbox_normalize_targets_precomputed) {
    FOR_RANGE(int32_t, i, 0, num_batch * num_roi_per_image) {
      bbox_targets_dptr[i * 4] = (bbox_targets_dptr[i * 4] - means_x) / stds_x;
      bbox_targets_dptr[i * 4 + 1] = (bbox_targets_dptr[i * 4 + 1] - means_y) / stds_y;
      bbox_targets_dptr[i * 4 + 2] = (bbox_targets_dptr[i * 4 + 2] - means_w) / stds_w;
      bbox_targets_dptr[i * 4 + 3] = (bbox_targets_dptr[i * 4 + 3] - means_h) / stds_h;
    }
  }
}
ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalTargetConf, ProposalTargetKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow