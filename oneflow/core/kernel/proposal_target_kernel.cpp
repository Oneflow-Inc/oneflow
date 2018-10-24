#include "oneflow/core/kernel/proposal_target_kernel.h"

namespace oneflow {

template<typename T>
void ProposalTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto gt_boxes = GetImageGtBoxes(BnInOp2Blob);
  auto roi_boxes = GetImageRoiBoxes(BnInOp2Blob);
  FindNearestGtBoxForEachRoiBox(BnInOp2Blob, gt_boxes, roi_boxes);
  SubsampleForegroundAndBackground(BnInOp2Blob, gt_boxes, roi_boxes);
  Output(BnInOp2Blob, gt_boxes, roi_boxes);
}

template<typename T>
typename ProposalTargetKernel<T>::LabeledGtBox ProposalTargetKernel<T>::GetImageGtBoxes(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  Blob* gt_labels_blob = BnInOp2Blob("gt_labels");
  Blob* gt_boxes_inds_blob = BnInOp2Blob("gt_boxes_inds");

  LabeledGtBox gt_boxes(GtBoxIndices(IndexSequence(gt_boxes_inds_blob->shape().elem_cnt(), 0,
                                                   gt_boxes_inds_blob->mut_dptr<int32_t>(), false),
                                     gt_boxes_blob->dptr<T>()),
                        gt_labels_blob->mut_dptr<int32_t>());
  FOR_RANGE(int32_t, i, 0, gt_boxes_blob->shape().At(0)) {
    int32_t dim1_num = gt_boxes_blob->shape().At(1);
    int32_t dim1_valid_num = gt_boxes_blob->dim1_valid_num(i);
    CHECK_EQ(dim1_valid_num, gt_labels_blob->dim1_valid_num(i));
    FOR_RANGE(int32_t, j, 0, dim1_valid_num) {
      if (GtBox::Cast(gt_boxes_blob->dptr<T>(i, j))->Area() > 0
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
  Blob* rois_inds_blob = BnInOp2Blob("rois_inds");

  return MaxOverlapOfRoiBoxWithGt(
      RoiBoxIndices(IndexSequence(rois_inds_blob->shape().elem_cnt(), rois_blob->dim0_valid_num(0),
                                  rois_inds_blob->mut_dptr<int32_t>(), true),
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
void ProposalTargetKernel<T>::ConcatGtBoxesToRoiBoxesHead(const LabeledGtBox& gt_boxes,
                                                          MaxOverlapOfRoiBoxWithGt& boxes) const {
  boxes.Truncate(boxes.size() + gt_boxes.size());
  for (int32_t i = boxes.size() - 1; i >= 0; --i) {
    boxes.index()[i + gt_boxes.size()] = boxes.GetIndex(i);
  }
  // Set gt box index in rois_boxes_inds to -(gt_index+1)
  // 0 -> -1  1 -> -2  2 -> -3
  FOR_RANGE(size_t, i, 0, gt_boxes.size()) { boxes.index()[i] = -(gt_boxes.GetIndex(i) + 1); }
}

template<typename T>
void ProposalTargetKernel<T>::SubsampleForegroundAndBackground(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob, const LabeledGtBox& gt_boxes,
    MaxOverlapOfRoiBoxWithGt& boxes) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  Blob* sampled_inds_blob = BnInOp2Blob("sampled_inds");
  size_t total_num_sampled_rois = sampled_inds_blob->shape().elem_cnt();
  IndexSequence sampled_inds(total_num_sampled_rois, 0, sampled_inds_blob->mut_dptr<int32_t>(),
                             false);

  boxes.Sort([&](int32_t lhs_index, int32_t rhs_index) {
    return boxes.max_overlap(lhs_index) > boxes.max_overlap(rhs_index);
  });
  ConcatGtBoxesToRoiBoxesHead(gt_boxes, boxes);
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
  boxes.Slice(0, bg_cnt).ForEach([&](int32_t index) {
    boxes.set_max_overlap_with_index(index, -1);
    return true;
  });
  sampled_inds.Concat(boxes.Slice(0, bg_cnt));
  // Return sampled inds
  boxes.Assign(sampled_inds);
}

template<typename T>
void ProposalTargetKernel<T>::Output(const std::function<Blob*(const std::string&)>& BnInOp2Blob,
                                     const LabeledGtBox& gt_boxes,
                                     const MaxOverlapOfRoiBoxWithGt& boxes) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  Blob* out_rois_blob = BnInOp2Blob("out_rois");
  Blob* labels_blob = BnInOp2Blob("labels");
  Blob* bbox_targets_blob = BnInOp2Blob("bbox_targets");
  Blob* bbox_inside_weights_blob = BnInOp2Blob("bbox_inside_weights");
  Blob* bbox_outside_weights_blob = BnInOp2Blob("bbox_outside_weights");
  std::memset(out_rois_blob->mut_dptr(), 0, out_rois_blob->shape().elem_cnt() * sizeof(T));
  std::memset(labels_blob->mut_dptr(), 0, labels_blob->shape().elem_cnt() * sizeof(int32_t));
  std::memset(bbox_targets_blob->mut_dptr(), 0, bbox_targets_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_inside_weights_blob->mut_dptr(), 0,
              bbox_inside_weights_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_outside_weights_blob->mut_dptr(), 0,
              bbox_outside_weights_blob->shape().elem_cnt() * sizeof(T));

  FOR_RANGE(size_t, i, 0, boxes.size()) {
    int32_t gt_index = boxes.GetMaxOverlapWithIndex(i);
    int32_t label = gt_index >= 0 ? gt_boxes.label(gt_index) : 0;
    labels_blob->mut_dptr<int32_t>()[i] = label;
    int32_t index = boxes.GetIndex(i);
    int32_t im_index = -1;
    auto* out_rois_bbox = BBox::Cast(out_rois_blob->mut_dptr<T>(i));
    auto* bbox_targets = BBoxDelta<T>::Cast(bbox_targets_blob->mut_dptr<T>(i));
    if (index >= 0) {
      const auto* rois_bbox = boxes.bbox(index);
      out_rois_bbox->set_corner_coord(rois_bbox->left(), rois_bbox->top(), rois_bbox->right(),
                                      rois_bbox->bottom());
      if (label > 0) {
        bbox_targets[label].TransformInverse(rois_bbox, gt_boxes.bbox(gt_index),
                                             conf.bbox_reg_weights());
      }
      im_index = rois_bbox->index();
    } else {
      int32_t index_gt = -index - 1;
      const auto* gt_bbox = gt_boxes.bbox(index_gt);
      out_rois_bbox->set_corner_coord(gt_bbox->left(), gt_bbox->top(), gt_bbox->right(),
                                      gt_bbox->bottom());
      if (label > 0) {
        bbox_targets[label].TransformInverse(gt_bbox, gt_boxes.bbox(gt_index),
                                             conf.bbox_reg_weights());
      }
      im_index = index_gt / BnInOp2Blob("gt_boxes")->shape().At(1);
    }
    out_rois_bbox->set_index(im_index);
    if (label > 0) {
      auto* inside_weights = BBoxWeights<T>::Cast(bbox_inside_weights_blob->mut_dptr<T>(i));
      inside_weights[label].set_weight_x(1.0);
      inside_weights[label].set_weight_y(1.0);
      inside_weights[label].set_weight_w(1.0);
      inside_weights[label].set_weight_h(1.0);
      auto* outside_weights = BBoxWeights<T>::Cast(bbox_outside_weights_blob->mut_dptr<T>(i));
      outside_weights[label].set_weight_x(1.0);
      outside_weights[label].set_weight_y(1.0);
      outside_weights[label].set_weight_w(1.0);
      outside_weights[label].set_weight_h(1.0);
    }
    if (BnInOp2Blob("rois")->has_record_id_in_device_piece_field()) {
      out_rois_blob->set_record_id_in_device_piece(i, im_index);
      labels_blob->set_record_id_in_device_piece(i, im_index);
      bbox_targets_blob->set_record_id_in_device_piece(i, im_index);
      bbox_inside_weights_blob->set_record_id_in_device_piece(i, im_index);
      bbox_outside_weights_blob->set_record_id_in_device_piece(i, im_index);
    }
  }
  out_rois_blob->set_dim0_valid_num(0, boxes.size());
  labels_blob->set_dim0_valid_num(0, boxes.size());
  bbox_targets_blob->set_dim0_valid_num(0, boxes.size());
  bbox_inside_weights_blob->set_dim0_valid_num(0, boxes.size());
  bbox_outside_weights_blob->set_dim0_valid_num(0, boxes.size());
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
