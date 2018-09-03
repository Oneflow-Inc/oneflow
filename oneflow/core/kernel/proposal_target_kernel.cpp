#include "oneflow/core/kernel/proposal_target_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

namespace {

template<typename T1, typename T2>
void CopyBoxes(const BBox<T2>* input, BBox<T1>* output) {
  output->set_x1(static_cast<T1>(input->x1()));
  output->set_y1(static_cast<T1>(input->y1()));
  output->set_x2(static_cast<T1>(input->x2()));
  output->set_y2(static_cast<T1>(input->y2()));
}

template<typename T1, typename T2, typename T3>
void ComputeBoxesDelta(const BBox<T2>* box, const BBox<T3>* target_box,
                       const BBoxRegressionWeights& bbox_reg_ws, BBoxDelta<T1>* delta) {
  delta->TransformInverse(box, target_box, bbox_reg_ws);
}

template<typename T>
void CopyToRoi(int32_t index, const BoxesSlice<T>& boxes, const GtBoxes<FloatList16>& gt_boxes,
               BBox<T>* roi_boxes) {
  if (index >= 0) {
    const BBox<T>* box = boxes.bbox(index);
    CopyBoxes(box, roi_boxes);
  } else {
    const BBox<float>* box = gt_boxes.GetBBox<float>(-index - 1);
    CopyBoxes(box, roi_boxes);
  }
}

template<typename T>
void ComputeBBoxTargets(int32_t index, int32_t gt_index, const BoxesSlice<T>& boxes,
                        const GtBoxes<FloatList16>& gt_boxes,
                        const BBoxRegressionWeights& bbox_reg_ws, BBoxDelta<T>* roi_bbox_target) {
  if (gt_index >= 0) {
    if (index >= 0) {
      const BBox<T>* box = boxes.bbox(index);
      const BBox<float>* gt_box = gt_boxes.GetBBox<float>(gt_index);
      ComputeBoxesDelta(box, gt_box, bbox_reg_ws, roi_bbox_target);
    } else {
      const BBox<float>* box = gt_boxes.GetBBox<float>(-index - 1);
      const BBox<float>* gt_box = gt_boxes.GetBBox<float>(gt_index);
      ComputeBoxesDelta(box, gt_box, bbox_reg_ws, roi_bbox_target);
    }
  }
}

}  // namespace

template<typename T>
void ProposalTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ClearOutputBlob(ctx, BnInOp2Blob);
  FOR_RANGE(int64_t, i, 0, BnInOp2Blob("rpn_rois")->shape().At(0)) {
    auto gt_boxes = GetImageGtBoxesWithLabels(i, BnInOp2Blob);
    auto roi_boxes = GetRoiBoxesSlice(i, BnInOp2Blob);
    auto boxes_max_overlap = ComputeRoiBoxesAndGtBoxesOverlaps(roi_boxes, gt_boxes, BnInOp2Blob);
    ConcatGtBoxesToRoiBoxes(gt_boxes, boxes_max_overlap);
    SubsampleForegroundAndBackground(boxes_max_overlap);
    ComputeAndWriteOutput(i, boxes_max_overlap, gt_boxes, BnInOp2Blob);
  }
}

template<typename T>
void ProposalTargetKernel<T>::ClearOutputBlob(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* labels_blob = BnInOp2Blob("labels");
  Blob* bbox_targets_blob = BnInOp2Blob("bbox_targets");
  Blob* bbox_inside_weights_blob = BnInOp2Blob("bbox_inside_weights");
  Blob* bbox_outside_weights_blob = BnInOp2Blob("bbox_outside_weights");
  std::memset(rois_blob->mut_dptr<T>(), 0, rois_blob->shape().elem_cnt() * sizeof(T));
  std::memset(labels_blob->mut_dptr<int32_t>(), 0,
              labels_blob->shape().elem_cnt() * sizeof(int32_t));
  std::memset(bbox_targets_blob->mut_dptr<T>(), 0,
              bbox_targets_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_inside_weights_blob->mut_dptr<T>(), 0,
              bbox_inside_weights_blob->shape().elem_cnt() * sizeof(T));
  std::memset(bbox_outside_weights_blob->mut_dptr<T>(), 0,
              bbox_outside_weights_blob->shape().elem_cnt() * sizeof(T));
}

template<typename T>
typename ProposalTargetKernel<T>::GtBoxesWithLabelsType
ProposalTargetKernel<T>::GetImageGtBoxesWithLabels(
    size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const FloatList16* gt_boxes = BnInOp2Blob("gt_boxes")->dptr<FloatList16>(im_index);
  const Int32List16* gt_labels = BnInOp2Blob("gt_labels")->dptr<Int32List16>(im_index);
  GtBoxesWithLabelsType gt_boxes_with_labels(*gt_boxes, *gt_labels);
  gt_boxes_with_labels.ConvertNormalToAbsCoord<float>(conf.image_height(), conf.image_width());
  CHECK_LE(gt_boxes_with_labels.size(), conf.max_gt_boxes_num());
  return gt_boxes_with_labels;
}

template<typename T>
BoxesSlice<T> ProposalTargetKernel<T>::GetRoiBoxesSlice(
    size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* boxes_index_blob = BnInOp2Blob("boxes_index");
  BoxesSlice<T> roi_boxes_slice(boxes_index_blob->shape().elem_cnt(),
                                boxes_index_blob->mut_dptr<int32_t>(),
                                BnInOp2Blob("rpn_rois")->dptr<T>(im_index));
  roi_boxes_slice.Truncate(BnInOp2Blob("rpn_rois")->shape().At(1));
  return roi_boxes_slice;
}

template<typename T>
typename ProposalTargetKernel<T>::BoxesWithMaxOverlapSlice
ProposalTargetKernel<T>::ComputeRoiBoxesAndGtBoxesOverlaps(
    const BoxesSlice<T>& roi_boxes, const GtBoxesType& gt_boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* max_overlaps_blob = BnInOp2Blob("max_overlaps");
  std::memset(max_overlaps_blob->mut_dptr(), 0,
              max_overlaps_blob->shape().elem_cnt() * sizeof(float));
  BoxesWithMaxOverlapSlice boxes_overlap_slice(
      roi_boxes, max_overlaps_blob->mut_dptr<float>(),
      BnInOp2Blob("max_overlaps_gt_boxes_index")->mut_dptr<int32_t>());
  FasterRcnnUtil<T>::ForEachOverlapBetweenBoxesAndGtBoxes(
      roi_boxes, gt_boxes, [&](int32_t index, int32_t gt_index, float overlap) {
        boxes_overlap_slice.UpdateMaxOverlapGtBox(index, gt_index, overlap);
      });
  return boxes_overlap_slice;
}

template<typename T>
void ProposalTargetKernel<T>::ConcatGtBoxesToRoiBoxes(const GtBoxesType& gt_boxes,
                                                      BoxesSlice<T>& roi_boxes) const {
  FOR_RANGE(size_t, i, 0, gt_boxes.size()) { roi_boxes.PushBack(-(i + 1)); }
}

template<typename T>
void ProposalTargetKernel<T>::SubsampleForegroundAndBackground(
    BoxesWithMaxOverlapSlice& boxes_max_overlap) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  int32_t fg_end = -1;
  int32_t bg_begin = -1;
  int32_t bg_end = -1;
  boxes_max_overlap.SortByOverlap(
      [](float lhs_overlap, float rhs_overlap) { return lhs_overlap > rhs_overlap; });
  boxes_max_overlap.ForEachOverlap([&](float overlap, size_t n, int32_t index) {
    if (overlap < conf.foreground_threshold() && fg_end == -1) { fg_end = n; }
    if (overlap < conf.background_threshold_high()) {
      if (bg_begin == -1) { bg_begin = n; }
      boxes_max_overlap.set_max_overlap_gt_box_index(index, -1);
    }
    if (overlap < conf.background_threshold_low()) {
      bg_end = n;
      return false;
    }
    return true;
  });
  if (fg_end <= 0) {
    boxes_max_overlap.Truncate(0);
    return;
  }
  if (bg_end == -1) { bg_end = boxes_max_overlap.size(); }
  CHECK_GE(bg_begin, fg_end);
  CHECK_GE(bg_end, bg_begin);

  size_t total_sample_cnt = conf.num_rois_per_image();
  size_t fg_cnt = total_sample_cnt * conf.foreground_fraction();
  size_t bg_cnt = 0;
  Slice sampled_slice(total_sample_cnt, boxes_max_overlap.mut_index_ptr(), false);
  sampled_slice.Truncate(0);
  if (fg_cnt < fg_end) {
    boxes_max_overlap.Shuffle(0, fg_end);
  } else {
    fg_cnt = fg_end;
  }
  sampled_slice.Concat(boxes_max_overlap.Sub(0, fg_cnt));
  bg_cnt = total_sample_cnt - fg_cnt;
  if (bg_cnt < bg_end - bg_begin) {
    boxes_max_overlap.Shuffle(bg_begin, bg_end);
  } else {
    bg_cnt = bg_end - bg_begin;
  }
  sampled_slice.Concat(boxes_max_overlap.Sub(bg_begin, bg_begin + bg_cnt));
  boxes_max_overlap.Fill(sampled_slice);
}

template<typename T>
void ProposalTargetKernel<T>::ComputeAndWriteOutput(
    size_t im_index, const BoxesWithMaxOverlapSlice& boxes, const GtBoxesWithLabelsType& gt_boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const BBoxRegressionWeights& bbox_reg_ws = conf.bbox_reg_weights();
  int64_t output_num = conf.num_rois_per_image();
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* labels_blob = BnInOp2Blob("labels");
  Blob* bbox_targets_blob = BnInOp2Blob("bbox_targets");
  Blob* bbox_inside_weights_blob = BnInOp2Blob("bbox_inside_weights");
  Blob* bbox_outside_weights_blob = BnInOp2Blob("bbox_outside_weights");

  BBox<T>* rois_bbox = BBox<T>::MutCast(rois_blob->mut_dptr<T>(im_index));
  FOR_RANGE(size_t, i, 0, boxes.size()) {
    int32_t index = boxes.GetIndex(i);
    int32_t gt_index = boxes.GetMaxOverlapGtBoxIndex(i);
    int32_t label = gt_boxes.GetLabel(gt_index);
    CopyToRoi(index, boxes, gt_boxes, rois_bbox + i);
    if (label > 0) {
      labels_blob->mut_dptr<int32_t>(im_index)[i] = label;
      int64_t bbox_offset = im_index * output_num + i;
      BBoxDelta<T>* bbox_targets =
          BBoxDelta<T>::MutCast(bbox_targets_blob->mut_dptr<T>(bbox_offset));
      ComputeBBoxTargets(index, gt_index, boxes, gt_boxes, bbox_reg_ws, bbox_targets + label);
      BBoxWeights<T>* inside_weights =
          BBoxWeights<T>::MutCast(bbox_inside_weights_blob->mut_dptr<T>(bbox_offset));
      BBoxWeights<T>* outside_weights =
          BBoxWeights<T>::MutCast(bbox_outside_weights_blob->mut_dptr<T>(bbox_offset));
      inside_weights[label].set_weight_x(1.0);
      inside_weights[label].set_weight_y(1.0);
      inside_weights[label].set_weight_w(1.0);
      inside_weights[label].set_weight_h(1.0);
      outside_weights[label].set_weight_x(1.0);
      outside_weights[label].set_weight_y(1.0);
      outside_weights[label].set_weight_w(1.0);
      outside_weights[label].set_weight_h(1.0);
    }
  }
}

template<typename T>
void ProposalTargetKernel<T>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("rois")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("rpn_rois"));
  BnInOp2Blob("bbox_targets")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("rpn_rois"));
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalTargetConf, ProposalTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
