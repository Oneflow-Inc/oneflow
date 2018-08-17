#include "oneflow/core/kernel/proposal_target_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

template<typename T>
void ProposalTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* rpn_rois_blob = BnInOp2Blob("rpn_rois");               //(im_num, roi, 4)
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");               // Pb (im_num)
  const Blob* gt_label_blob = BnInOp2Blob("gt_label");               // Pb (im_num)
  Blob* rois_blob = BnInOp2Blob("rois");                             //(im_num, sample_num, 4)
  Blob* labels_blob = BnInOp2Blob("labels");                         //(im_num, sample_num, 1)
  Blob* bbox_targets_blob = BnInOp2Blob("bbox_targets");             //(im_num, sample_num, 4*class)
  Blob* inside_weights_blob = BnInOp2Blob("bbox_inside_weights");    //(im_num, sample_num, 4*class)
  Blob* outside_weights_blob = BnInOp2Blob("bbox_outside_weights");  //(im_num, sample_num, 4*class)
  // tmp blob
  int32_t* roi_nearest_gt_index_ptr =
      BnInOp2Blob("roi_nearest_gt_index")->mut_dptr<int32_t>();                          //(roi)
  T* roi_max_overlap_ptr = BnInOp2Blob("roi_max_overlap")->mut_dptr<T>();                //(roi)
  int32_t* rois_index_ptr = BnInOp2Blob("rois_index")->mut_dptr<int32_t>();              //(roi)
  FloatList16* gt_boxes_tmp_ptr = BnInOp2Blob("gt_boxes_tmp")->mut_dptr<FloatList16>();  //(gt_num)
  int64_t im_num = rpn_rois_blob->shape().At(0);
  int64_t roi_num = rpn_rois_blob->shape().At(1);

  FOR_RANGE(int64_t, i, 0, im_num) {
    const T* rpn_rois_ptr = rpn_rois_blob->dptr<T>(i);
    const FloatList16* gt_boxes_ptr = gt_boxes_blob->dptr<FloatList16>(i);
    const Int32List16* gt_labels_ptr = gt_label_blob->dptr<Int32List16>(i);
    T* rois_ptr = rois_blob->mut_dptr<T>(i);
    int32_t* labels_ptr = labels_blob->mut_dptr<int32_t>(i);
    T* bbox_targets_ptr = bbox_targets_blob->mut_dptr<T>(i);
    T* inside_weights_ptr = inside_weights_blob->mut_dptr<T>(i);
    T* outside_weights_ptr = outside_weights_blob->mut_dptr<T>(i);
    FOR_RANGE(int32_t, i, 0, gt_boxes_ptr->value().value_size()) {
      // gt_boxes_tmp_ptr->mutable_value()->set_value(i, gt_boxes_ptr->value().value(i) * 720);
      gt_boxes_tmp_ptr->mutable_value()->add_value(gt_boxes_ptr->value().value(i) * 720);
    }

    RoisNearestGtAndMaxIou(roi_num, rpn_rois_ptr, gt_boxes_tmp_ptr, roi_nearest_gt_index_ptr,
                           roi_max_overlap_ptr);
    ScoredBBoxSlice<T> rois_slice(roi_num, rpn_rois_ptr, roi_max_overlap_ptr, rois_index_ptr);
    rois_slice.DescSortByScore();
    ScoredBBoxSlice<T> fg_slice = ForegroundChoice(rois_slice);
    ScoredBBoxSlice<T> bg_slice = BackgroundChoice(rois_slice, fg_slice.available_len());
    ComputeTargetAndWriteOut(fg_slice, bg_slice, roi_nearest_gt_index_ptr, gt_boxes_tmp_ptr,
                             gt_labels_ptr, rois_ptr, labels_ptr, bbox_targets_ptr,
                             inside_weights_ptr, outside_weights_ptr);
    gt_boxes_tmp_ptr->mutable_value()->clear_value();
  }
}

template<typename T>
void ProposalTargetKernel<T>::RoisNearestGtAndMaxIou(const int64_t rois_num, const T* rpn_rois_ptr,
                                                     const FloatList16* gt_boxes_tmp_ptr,
                                                     int32_t* roi_nearest_gt_index_ptr,
                                                     T* roi_max_overlap_ptr) const {
  const BBox<T>* roi_box = BBox<T>::Cast(rpn_rois_ptr);
  const BBox<float>* gt_bbox = BBox<float>::Cast(gt_boxes_tmp_ptr->value().value().data());
  const int64_t gt_num = gt_boxes_tmp_ptr->value().value_size() / 4;
  FOR_RANGE(int64_t, i, 0, rois_num) {
    float maxIou = 0;
    int64_t maxIouIdx = 0;
    FOR_RANGE(int64_t, j, 0, gt_num) {
      float iou = roi_box[i].InterOverUniontmp(&gt_bbox[j]);
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
ScoredBBoxSlice<T> ProposalTargetKernel<T>::ForegroundChoice(ScoredBBoxSlice<T>& rois_slice) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const int64_t fg_end_index = rois_slice.FindByThreshold(conf.fg_thresh());
  ScoredBBoxSlice<T> fg_rois_slice = rois_slice.Slice(0, fg_end_index);
  const int64_t num_roi_per_image = conf.num_roi_per_image();
  const float fg_fraction = conf.fg_fraction();
  const int64_t fg_sample_size =
      std::min(fg_end_index + 1, (int64_t)std::round(num_roi_per_image * fg_fraction));
  if (fg_sample_size < fg_end_index + 1) {
    fg_rois_slice.Shuffle();
    fg_rois_slice.Truncate(fg_sample_size);
  }
  return fg_rois_slice;
}

template<typename T>
ScoredBBoxSlice<T> ProposalTargetKernel<T>::BackgroundChoice(ScoredBBoxSlice<T>& rois_slice,
                                                             const int64_t fg_sample_size) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  CHECK_GT(conf.bg_thresh_hi(), conf.bg_thresh_lo());
  int64_t bg_start_index = rois_slice.FindByThreshold(conf.bg_thresh_hi());
  int64_t bg_end_index = rois_slice.FindByThreshold(conf.bg_thresh_lo());
  ScoredBBoxSlice<T> bg_rois_slice = rois_slice.Slice(bg_start_index, bg_end_index);
  const int64_t num_roi_per_image = conf.num_roi_per_image();
  const int64_t bg_sample_size =
      std::min(bg_end_index - bg_start_index + 1, num_roi_per_image - fg_sample_size);
  if (bg_sample_size < bg_end_index - bg_start_index + 1) {
    bg_rois_slice.Shuffle();
    bg_rois_slice.Truncate(bg_sample_size);
  }
  return bg_rois_slice;
}

template<typename T>
void ProposalTargetKernel<T>::CopyRoIs(const ScoredBBoxSlice<T>& slice, T* rois_ptr) const {
  FOR_RANGE(int32_t, i, 0, slice.available_len()) {
    BBox<T>* rois_box = BBox<T>::MutCast(rois_ptr);
    const BBox<T>* bbox = slice.GetBBox(i);
    rois_box[i].set_x1(bbox->x1());
    rois_box[i].set_y1(bbox->y1());
    rois_box[i].set_x2(bbox->x2());
    rois_box[i].set_y2(bbox->y2());
  }
}

template<typename T>
void ProposalTargetKernel<T>::ComputeTargetAndWriteOut(
    const ScoredBBoxSlice<T>& fg_slice, const ScoredBBoxSlice<T>& bg_slice,
    const int32_t* roi_nearest_gt_index_ptr, const FloatList16* gt_boxes_ptr,
    const Int32List16* gt_labels_ptr, T* rois_ptr, int32_t* labels_ptr, T* bbox_targets_ptr,
    T* inside_weights_ptr, T* outside_weights_ptr) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const BBox<float>* gt_bbox = BBox<float>::Cast(gt_boxes_ptr->value().value().data());
  BBoxDelta<T>* bbox_targets_box = BBoxDelta<T>::MutCast(bbox_targets_ptr);
  BBoxWeights<T>* inside_weights_box = BBoxWeights<T>::MutCast(inside_weights_ptr);
  BBoxWeights<T>* outside_weights_box = BBoxWeights<T>::MutCast(outside_weights_ptr);
  const BBoxRegressionWeights& bbox_reg_ws = conf.bbox_reg_weights();
  const int32_t class_num = conf.class_num();
  const BboxWeightConf& bbox_inside_weight_conf = conf.bbox_inside_weight_conf();

  CopyRoIs(fg_slice, rois_ptr);
  CopyRoIs(bg_slice, rois_ptr + 4 * fg_slice.available_len());

  FOR_RANGE(int64_t, i, 0, fg_slice.available_len()) {
    const BBox<T>* bbox = fg_slice.GetBBox(i);
    const int32_t roi_index = fg_slice.GetSlice(i);
    const int32_t gt_index = roi_nearest_gt_index_ptr[roi_index];
    const int64_t target_index = i * class_num + gt_labels_ptr->value().value(gt_index);
    labels_ptr[i] = gt_labels_ptr->value().value(gt_index);
    bbox_targets_box[target_index].TransformInversetmp(bbox, &gt_bbox[gt_index], bbox_reg_ws);
    inside_weights_box[target_index].set_weight_x(bbox_inside_weight_conf.weight_x());
    inside_weights_box[target_index].set_weight_y(bbox_inside_weight_conf.weight_y());
    inside_weights_box[target_index].set_weight_w(bbox_inside_weight_conf.weight_w());
    inside_weights_box[target_index].set_weight_h(bbox_inside_weight_conf.weight_h());
    outside_weights_box[target_index].set_weight_x(bbox_inside_weight_conf.weight_x() > 0 ? 1 : 0);
    outside_weights_box[target_index].set_weight_y(bbox_inside_weight_conf.weight_y() > 0 ? 1 : 0);
    outside_weights_box[target_index].set_weight_w(bbox_inside_weight_conf.weight_w() > 0 ? 1 : 0);
    outside_weights_box[target_index].set_weight_h(bbox_inside_weight_conf.weight_h() > 0 ? 1 : 0);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalTargetConf, ProposalTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
