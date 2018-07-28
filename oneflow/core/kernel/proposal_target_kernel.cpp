//#include "oneflow/core/kernel/proposal_kernel.h"
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
    all_rois_dptr[all_rois_idx + 0] = gt_box_dptr[i * 5 + 0];
    all_rois_dptr[all_rois_idx + 1] = gt_box_dptr[i * 5 + 1];
    all_rois_dptr[all_rois_idx + 2] = gt_box_dptr[i * 5 + 2];
    all_rois_dptr[all_rois_idx + 3] = gt_box_dptr[i * 5 + 3];
  }
}
template<typename T>
void BboxTransform(int64_t m, const T* bbox, const T* target_bbox, T* deltas) {
  for (int64_t i = 0; i < m; ++i) {
    float b_w = bbox[i * 4 + 2] - bbox[i * 4] + 1.0f;
    float b_h = bbox[i * 4 + 3] - bbox[i * 4 + 1] + 1.0f;
    float b_ctr_x = bbox[i * 4] + 0.5f * b_w;
    float b_ctr_y = bbox[i * 4 + 1] + 0.5f * b_h;

    float t_w = target_bbox[i * 4 + 2] - target_bbox[i * 4] + 1.0f;
    float t_h = target_bbox[i * 4 + 3] - target_bbox[i * 4 + 1] + 1.0f;
    float t_ctr_x = target_bbox[i * 4] + 0.5f * t_w;
    float t_ctr_y = target_bbox[i * 4 + 1] + 0.5f * t_h;

    deltas[i * 4 + 0] = (t_ctr_x - b_ctr_x) / b_w;
    deltas[i * 4 + 1] = (t_ctr_y - b_ctr_y) / b_h;
    deltas[i * 4 + 2] = std::log(t_w / b_w);
    deltas[i * 4 + 3] = std::log(t_h / b_h);
  }
}
template<typename T>
void BboxNormalize(int64_t m, const BboxNormalizeMeansConf& means_conf,
                   const BboxNormalizeStdsConf& stds_conf, const T* bbox_targets_dptr,
                   T* bbox_targets_normalized) {
  const int32_t means_x = means_conf.means_x();
  const int32_t means_y = means_conf.means_y();
  const int32_t means_w = means_conf.means_w();
  const int32_t means_h = means_conf.means_h();
  const int32_t stds_x = stds_conf.stds_x();
  const int32_t stds_y = stds_conf.stds_y();
  const int32_t stds_w = stds_conf.stds_w();
  const int32_t stds_h = stds_conf.stds_h();
  FOR_RANGE(int32_t, i, 0, m) {
    bbox_targets_normalized[i * 4 + 0] = (bbox_targets_dptr[i * 4 + 0] - means_x) / stds_x;
    bbox_targets_normalized[i * 4 + 1] = (bbox_targets_dptr[i * 4 + 1] - means_y) / stds_y;
    bbox_targets_normalized[i * 4 + 2] = (bbox_targets_dptr[i * 4 + 2] - means_w) / stds_w;
    bbox_targets_normalized[i * 4 + 3] = (bbox_targets_dptr[i * 4 + 3] - means_h) / stds_h;
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
  Memset<DeviceType::kCPU>(ctx.device_ctx, rois_blob->mut_dptr<T>(), 0,
                           rois_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, labels_blob->mut_dptr<int32_t>(), 0,
                           labels_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, bbox_targets_blob->mut_dptr<T>(), 0,
                           bbox_targets_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, inside_weights_blob->mut_dptr<T>(), 0,
                           inside_weights_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, outside_weights_blob->mut_dptr<T>(), 0,
                           outside_weights_blob->ByteSizeOfDataContentField());
  // data_tmp blob
  Blob* all_rois_blob = BnInOp2Blob("all_rois");
  Blob* overlap_blob = BnInOp2Blob("bbox_overlap");
  Blob* roi_argmax_blob = BnInOp2Blob("roi_argmax");
  Blob* max_overlaps_blob = BnInOp2Blob("max_overlaps");
  Blob* fg_inds_blob = BnInOp2Blob("fg_inds");
  Blob* bg_inds_blob = BnInOp2Blob("bg_inds");
  Blob* fg_bg_sample_inds_blob = BnInOp2Blob("fg_bg_sample_inds");
  Blob* bbox_target_data_blob = BnInOp2Blob("bbox_target_data");
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
  const int32_t class_num = this->op_conf().proposal_target_conf().class_num();
  int32_t num_batch = rpn_rois_blob->shape().At(0);
  int32_t rois_num = rpn_rois_blob->shape().At(1);
  int32_t gt_max_num = gt_boxes_blob->shape().At(1);
  const T* rpn_rois_dptr = rpn_rois_blob->dptr<T>();  //(batch_num,roi_num,4)
  const T* gt_bbox_dptr = gt_boxes_blob->dptr<T>();   //(batch_num,gt_max_num,5)
  const T* gt_num_dptr = gt_num_blob->dptr<T>();      //(batch_num,1)
  T* labels_dptr =
      labels_blob->mut_dptr<int32_t>();           //(batch_num * num_roi_per_image,1)  1 or num_class???
  T* rois_dptr = rois_blob->mut_dptr<T>();  //(batch_num , num_roi_per_image, 4)
  T* bbox_targets_dptr =
      bbox_targets_blob->mut_dptr<T>();  //(batch_num * num_roi_per_image,num_class*4)
  T* inside_weights_dptr =
      inside_weights_blob->mut_dptr<T>();  //(batch_num * num_roi_per_image,num_class*4)
  T* outside_weights_dptr =
      outside_weights_blob->mut_dptr<T>();  //(batch_num * num_roi_per_image,num_class*4)
  // tmp blob
  T* all_rois_dptr = all_rois_blob->mut_dptr<T>();  // roi_num = roi_num + gt_max_num(roi_num, 4)
  T* overlap_dptr = overlap_blob->mut_dptr<T>();    //(roi_num,gt_max_num)
  T* roi_argmax_dptr = roi_argmax_blob->mut_dptr<int32_t>();                //(roi_num,1)
  T* max_overlaps_dptr = max_overlaps_blob->mut_dptr<T>();            //(roi_num,1)
  T* fg_inds_dptr = fg_inds_blob->mut_dptr<int32_t>();                      //(roi_num,1)
  T* bg_inds_dptr = bg_inds_blob->mut_dptr<int32_t>();                      //(roi_num,1)
  T* fg_bg_sample_inds_dptr = fg_bg_sample_inds_blob->mut_dptr<int32_t>();  //(num_roi_per_image,1)
  T* bbox_target_data_dptr = bbox_target_data_blob->mut_dptr<T>();    //(num_roi_per_image,4)
  int32_t out_weight_x = (weight_x > 0 ? weight_x : 0);
  int32_t out_weight_y = (weight_y > 0 ? weight_y : 0);
  int32_t out_weight_w = (weight_w > 0 ? weight_w : 0);
  int32_t out_weight_h = (weight_h > 0 ? weight_h : 0);
  FOR_RANGE(int32_t, i, 0, num_batch) {
    // 1. Include ground - truth boxes in rois
    // 2. overlaps: (rois x gt_boxes)
    // 3. get gt_assignment(argmax)&max_overlaps(max) for overlaps
    // 4. label=gt[gt_assignment,4]
    // 5. Select foreground RoIs indexs as maxoverlaps>FG_THRESH
    // 6. Sample foreground regions
    // 7. Select background RoIs indexs as [BG_THRESH_LO, BG_THRESH_HI)
    // 8. Compute number of background RoIs to take from this image
    // 9. Sample background regions without replacement
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
    rois_num = rois_num + gt_max_num;
    RcnnUtil<T>::BboxOverlaps(all_rois_dptr, rois_num, gt_bbox_dptr, gt_num, gt_max_num,
                              overlap_dptr);
    RcnnUtil<T>::OverlapRowArgMax7Max(overlap_dptr, rois_num, gt_num, gt_max_num, roi_argmax_dptr,
                                      max_overlaps_dptr);
    int32_t fgidx = 0;
    int32_t bgidx = 0;
    FOR_RANGE(int32_t, j, 0, rois_num) {
      if (max_overlaps_dptr[j] > fg_thresh) { fg_inds_dptr[fgidx++] = j; }
      if (max_overlaps_dptr[j] >= bg_thresh_lo && max_overlaps_dptr[j] < bg_thresh_hi) {
        bg_inds_dptr[bgidx++] = j;
      }
    }
    int32_t fg_inds_num = fgidx;
    int32_t bg_inds_num = bgidx;
    int32_t fg_rois_per_image =
        std::min(fg_inds_num, (int32_t)std::round(num_roi_per_image * fg_fraction));
    if (fg_inds_num > 0) {
      RcnnUtil<T>::SampleChoice(fg_inds_dptr, fg_inds_num, fg_bg_sample_inds_dptr,
                                fg_rois_per_image);
    }
    int32_t bg_rois_per_image = std::min(bg_inds_num, num_roi_per_image - fg_rois_per_image);
    if (bg_inds_num > 0) {
      RcnnUtil<T>::SampleChoice(bg_inds_dptr, bg_inds_num,
                                fg_bg_sample_inds_dptr + fg_rois_per_image, bg_rois_per_image);
    }
    FOR_RANGE(int32_t, j, 0, num_roi_per_image) {
      int32_t index = fg_bg_sample_inds_dptr[j];
      rois_dptr[j * 4 + 0] = all_rois_dptr[index * 4 + 0];
      rois_dptr[j * 4 + 1] = all_rois_dptr[index * 4 + 1];
      rois_dptr[j * 4 + 2] = all_rois_dptr[index * 4 + 2];
      rois_dptr[j * 4 + 3] = all_rois_dptr[index * 4 + 3];
      if (j < fg_rois_per_image)  // foreground
      {
        int32_t gt_index = roi_argmax_dptr[index];
        bbox_target_data_dptr[j * 4 + 0] = gt_bbox_dptr[gt_index * 5 + 0];
        bbox_target_data_dptr[j * 4 + 1] = gt_bbox_dptr[gt_index * 5 + 1];
        bbox_target_data_dptr[j * 4 + 2] = gt_bbox_dptr[gt_index * 5 + 2];
        bbox_target_data_dptr[j * 4 + 3] = gt_bbox_dptr[gt_index * 5 + 3];
        labels_dptr[j] = gt_bbox_dptr[gt_index * 5 + 4];
      }
    }
    BboxTransform(fg_rois_per_image, rois_dptr, bbox_target_data_dptr, bbox_target_data_dptr);
    if (bbox_normalize_targets_precomputed) {
      BboxNormalizeMeansConf means_conf =
          this->op_conf().proposal_target_conf().bbox_normalize_means();
      BboxNormalizeStdsConf stds_conf =
          this->op_conf().proposal_target_conf().bbox_normalize_stds();
      BboxNormalize(fg_rois_per_image, means_conf, stds_conf, bbox_target_data_dptr,
                    bbox_target_data_dptr);
    }
    FOR_RANGE(int32_t, j, 0, fg_rois_per_image) {
      int startidx = j * class_num * 4 + labels_dptr[j] * 4;
      bbox_targets_dptr[startidx + 0] = bbox_target_data_dptr[j * 4 + 0];
      bbox_targets_dptr[startidx + 1] = bbox_target_data_dptr[j * 4 + 1];
      bbox_targets_dptr[startidx + 2] = bbox_target_data_dptr[j * 4 + 2];
      bbox_targets_dptr[startidx + 3] = bbox_target_data_dptr[j * 4 + 3];
      inside_weights_dptr[startidx + 0] = weight_x;
      inside_weights_dptr[startidx + 1] = weight_y;
      inside_weights_dptr[startidx + 2] = weight_w;
      inside_weights_dptr[startidx + 3] = weight_h;
      outside_weights_dptr[startidx + 0] = out_weight_x;
      outside_weights_dptr[startidx + 1] = out_weight_y;
      outside_weights_dptr[startidx + 2] = out_weight_w;
      outside_weights_dptr[startidx + 3] = out_weight_h;
    }
  }
}
ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalTargetConf, ProposalTargetKernel,
                               ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
