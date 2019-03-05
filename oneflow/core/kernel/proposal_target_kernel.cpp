#include "oneflow/core/kernel/proposal_target_kernel.h"

namespace oneflow {

template<typename T>
void ProposalTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializeOutputBlob(ctx.device_ctx, BnInOp2Blob);
  auto gt_boxes = GetImageGtBoxes(BnInOp2Blob);
  auto roi_boxes = GetImageRoiBoxes(BnInOp2Blob);
  FindNearestGtBoxForEachRoiBox(BnInOp2Blob, gt_boxes, roi_boxes);
  SubsampleForegroundAndBackground(BnInOp2Blob, gt_boxes, roi_boxes);
  Output(BnInOp2Blob, gt_boxes, roi_boxes);
}

template<typename T>
void ProposalTargetKernel<T>::InitializeOutputBlob(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* sampled_rois_blob = BnInOp2Blob("sampled_rois");
  Blob* sampled_roi_inds_blob = BnInOp2Blob("sampled_roi_inds");
  Blob* class_labels_blob = BnInOp2Blob("class_labels");
  Blob* regression_targets_blob = BnInOp2Blob("regression_targets");
  Blob* regression_weights_blob = BnInOp2Blob("regression_weights");
  std::memset(sampled_rois_blob->mut_dptr(), 0,
              sampled_rois_blob->static_shape().elem_cnt() * sizeof(T));
  std::memset(sampled_roi_inds_blob->mut_dptr(), 0,
              sampled_roi_inds_blob->static_shape().elem_cnt() * sizeof(int32_t));
  std::memset(class_labels_blob->mut_dptr(), 0,
              class_labels_blob->static_shape().elem_cnt() * sizeof(int32_t));
  std::memset(regression_targets_blob->mut_dptr(), 0,
              regression_targets_blob->static_shape().elem_cnt() * sizeof(T));
  std::memset(regression_weights_blob->mut_dptr(), 0,
              regression_weights_blob->static_shape().elem_cnt() * sizeof(T));
}

template<typename T>
typename ProposalTargetKernel<T>::LabeledGtBox ProposalTargetKernel<T>::GetImageGtBoxes(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  Blob* gt_labels_blob = BnInOp2Blob("gt_labels");
  Blob* gt_box_inds_blob = BnInOp2Blob("gt_box_inds");

  LabeledGtBox gt_boxes(GtBoxIndices(IndexSequence(gt_box_inds_blob->shape().elem_cnt(), 0,
                                                   gt_box_inds_blob->mut_dptr<int32_t>(), false),
                                     gt_boxes_blob->dptr<T>()),
                        gt_labels_blob->mut_dptr<int32_t>());
  FOR_RANGE(int32_t, i, 0, gt_boxes_blob->shape().At(0)) {
    int32_t dim1_num = gt_boxes_blob->shape().At(1);
    int32_t dim1_valid_num = gt_boxes_blob->dim1_valid_num(i);
    CHECK_EQ(dim1_valid_num, gt_labels_blob->dim1_valid_num(i));
    FOR_RANGE(int32_t, j, 0, dim1_valid_num) {
      if (GtBBox::Cast(gt_boxes_blob->dptr<T>(i, j))->Area() > 0
          && *(gt_labels_blob->dptr<int32_t>(i, j)) <= conf.num_classes()) {
        gt_boxes.PushBack(i * dim1_num + j);
      }
    }
  }
  return gt_boxes;
}

template<typename T>
typename ProposalTargetKernel<T>::MaxOverlapOfRoiBoxWithGt
ProposalTargetKernel<T>::GetImageRoiBoxes(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* rois_blob = BnInOp2Blob("rois");
  Blob* roi_inds_blob = BnInOp2Blob("roi_inds");

  return MaxOverlapOfRoiBoxWithGt(
      RoiBoxIndices(IndexSequence(rois_blob->static_shape().At(0), rois_blob->dim0_valid_num(0),
                                  roi_inds_blob->mut_dptr<int32_t>(), true),
                    rois_blob->dptr<T>()),
      BnInOp2Blob("max_overlaps")->mut_dptr<float>(),
      BnInOp2Blob("max_overlaps_with_gt_index")->mut_dptr<int32_t>(), true);
}

template<typename T>
void ProposalTargetKernel<T>::FindNearestGtBoxForEachRoiBox(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob, const LabeledGtBox& gt_boxes,
    MaxOverlapOfRoiBoxWithGt& roi_boxes) const {
  FOR_RANGE(size_t, i, 0, roi_boxes.size()) {
    const auto* bbox = roi_boxes.GetBBox(i);
    FOR_RANGE(size_t, j, 0, gt_boxes.size()) {
      int32_t gt_index = gt_boxes.GetIndex(j);
      int32_t gt_im_index = gt_index / BnInOp2Blob("gt_boxes")->shape().At(1);
      if (gt_im_index == bbox->index()) {
        float overlap = bbox->InterOverUnion(gt_boxes.GetBBox(j));
        roi_boxes.TryUpdateMaxOverlap(roi_boxes.GetIndex(i), gt_index, overlap);
      }
    }
  }
}

template<typename T>
void ProposalTargetKernel<T>::SubsampleForegroundAndBackground(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob, const LabeledGtBox& gt_boxes,
    MaxOverlapOfRoiBoxWithGt& boxes) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  Blob* sampled_roi_inds_blob = BnInOp2Blob("sampled_roi_inds");
  size_t total_num_sampled_rois = sampled_roi_inds_blob->static_shape().elem_cnt();
  IndexSequence sampled_inds(total_num_sampled_rois, 0, sampled_roi_inds_blob->mut_dptr<int32_t>(),
                             false);

  boxes.Sort([&](int32_t lhs_index, int32_t rhs_index) {
    return boxes.max_overlap(lhs_index) > boxes.max_overlap(rhs_index);
  });
  // Foregroud sample
  size_t fg_end = boxes.Find(
      [&](int32_t index) { return boxes.max_overlap(index) < conf.foreground_threshold(); });
  size_t fg_cnt = total_num_sampled_rois * conf.foreground_fraction();
  if (fg_cnt < fg_end) {
    if (conf.random_subsample()) { boxes.Shuffle(0, fg_end); }
  } else {
    fg_cnt = fg_end;
  }
  sampled_inds.Concat(boxes.Slice(0, fg_cnt));
  boxes.Assign(boxes.Slice(fg_end, boxes.size()));

  // Backgroud sample
  boxes.Filter(
      [&](int32_t index) { return boxes.max_overlap(index) >= conf.background_threshold_high(); });
  size_t bg_end = boxes.Find(
      [&](int32_t index) { return boxes.max_overlap(index) < conf.background_threshold_low(); });
  size_t bg_cnt = total_num_sampled_rois - fg_cnt;
  if (bg_cnt < bg_end) {
    if (conf.random_subsample()) { boxes.Shuffle(0, bg_end); }
  } else {
    bg_cnt = bg_end;
  }
  // Set background box relative gt box index to -1
  // along with the relative label to be 0 to indicate
  // negative sample
  auto neg_sampled_inds = boxes.Slice(0, bg_cnt);
  neg_sampled_inds.ForEach([&](int32_t index) {
    boxes.set_max_overlap_with_index(index, -1);
    return true;
  });
  sampled_inds.Concat(neg_sampled_inds);
  boxes.Assign(sampled_inds);
}

template<typename T>
void ProposalTargetKernel<T>::Output(const std::function<Blob*(const std::string&)>& BnInOp2Blob,
                                     const LabeledGtBox& gt_boxes,
                                     const MaxOverlapOfRoiBoxWithGt& boxes) const {
  Blob* sampled_rois_blob = BnInOp2Blob("sampled_rois");
  Blob* sampled_roi_inds_blob = BnInOp2Blob("sampled_roi_inds");
  Blob* class_labels_blob = BnInOp2Blob("class_labels");
  Blob* regression_targets_blob = BnInOp2Blob("regression_targets");
  Blob* regression_weights_blob = BnInOp2Blob("regression_weights");
  auto* sampled_rois_bboxes = BBox::Cast(sampled_rois_blob->mut_dptr<T>());
  FOR_RANGE(size_t, i, 0, boxes.size()) {
    const int32_t index = boxes.GetIndex(i);
    const auto* rois_bbox = boxes.bbox(index);
    const int32_t im_index = rois_bbox->index();
    sampled_rois_bboxes[i].set_ltrb(rois_bbox->left(), rois_bbox->top(), rois_bbox->right(),
                                    rois_bbox->bottom());
    sampled_rois_bboxes[i].set_index(im_index);

    const int32_t gt_index = boxes.GetMaxOverlapWithIndex(i);
    const int32_t label = gt_index >= 0 ? gt_boxes.label(gt_index) : 0;
    class_labels_blob->mut_dptr<int32_t>()[i] = label;
    auto* bbox_regression = BBoxDelta<T>::Cast(regression_targets_blob->mut_dptr<T>());
    auto* regression_weights = BBoxWeights<T>::Cast(regression_weights_blob->mut_dptr<T>());
    if (label > 0) {
      bbox_regression[i].TransformInverse(rois_bbox, gt_boxes.bbox(gt_index),
                                          op_conf().proposal_target_conf().bbox_reg_weights());
      regression_weights[i].set_weight_x(OneVal<T>::value);
      regression_weights[i].set_weight_y(OneVal<T>::value);
      regression_weights[i].set_weight_w(OneVal<T>::value);
      regression_weights[i].set_weight_h(OneVal<T>::value);
    }
    if (BnInOp2Blob("rois")->has_record_id_in_device_piece_field()) {
      sampled_rois_blob->set_record_id_in_device_piece(i, im_index);
      sampled_roi_inds_blob->set_record_id_in_device_piece(i, im_index);
      class_labels_blob->set_record_id_in_device_piece(i, im_index);
      regression_targets_blob->set_record_id_in_device_piece(i, im_index);
      regression_weights_blob->set_record_id_in_device_piece(i, im_index);
    }
  }
  sampled_rois_blob->set_dim0_valid_num(0, boxes.size());
  sampled_roi_inds_blob->set_dim0_valid_num(0, boxes.size());
  class_labels_blob->set_dim0_valid_num(0, boxes.size());
  regression_targets_blob->set_dim0_valid_num(0, boxes.size());
  regression_weights_blob->set_dim0_valid_num(0, boxes.size());
}

template<typename T>
void ProposalTargetKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void ProposalTargetKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalTargetConf, ProposalTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
