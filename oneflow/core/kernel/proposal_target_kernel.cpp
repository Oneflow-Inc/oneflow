#include "oneflow/core/kernel/proposal_target_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

namespace {

template<typename T1, typename T2>
void CopyBBox(const BBox<T2>* input, BBox<T1>* output) {
  output->set_x1(static_cast<T1>(input->x1()));
  output->set_y1(static_cast<T1>(input->y1()));
  output->set_x2(static_cast<T1>(input->x2()));
  output->set_y2(static_cast<T1>(input->y2()));
}

template<typename T1, typename T2, typename T3>
void BBoxTransformInverse(const BBox<T2>* box, const BBox<T3>* target_box,
                          const BBoxRegressionWeights& bbox_reg_ws, BBoxDelta<T1>* delta) {
  delta->TransformInverse(box, target_box, bbox_reg_ws);
}

template<typename T>
void CopySampledBoxesToRois(int32_t index, const BoxesIndex<T>& boxes, const GtBoxes& gt_boxes,
                            BBox<T>* roi_boxes) {
  if (index >= 0) {
    const BBox<T>* box = boxes.bbox(index);
    CopyBBox(box, roi_boxes);
  } else {
    const BBox<float>* box = gt_boxes.GetBBox<float>(-index - 1);
    CopyBBox(box, roi_boxes);
  }
}

template<typename T>
void ComputeBBoxTargets(int32_t index, int32_t gt_index, const BoxesIndex<T>& boxes,
                        const GtBoxes& gt_boxes, const BBoxRegressionWeights& bbox_reg_ws,
                        BBoxDelta<T>* roi_bbox_target) {
  if (gt_index >= 0) {
    if (index >= 0) {
      const BBox<T>* box = boxes.bbox(index);
      const BBox<float>* gt_box = gt_boxes.GetBBox<float>(gt_index);
      BBoxTransformInverse(box, gt_box, bbox_reg_ws, roi_bbox_target);
    } else {
      const BBox<float>* box = gt_boxes.GetBBox<float>(-index - 1);
      const BBox<float>* gt_box = gt_boxes.GetBBox<float>(gt_index);
      BBoxTransformInverse(box, gt_box, bbox_reg_ws, roi_bbox_target);
    }
  }
}

}  // namespace

template<typename T>
void ProposalTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ClearOutputBlob(ctx, BnInOp2Blob);
  FOR_RANGE(int64_t, i, 0, BnInOp2Blob("rpn_rois")->shape().At(0)) {
    auto gt_boxes = GetImageGtBoxes(i, BnInOp2Blob);
    auto boxes = GetImageRoiBoxes(i, BnInOp2Blob);
    ComputeRoiBoxesAndGtBoxesOverlaps(gt_boxes, boxes);
    SubsampleForegroundAndBackground(gt_boxes, boxes);
    ComputeAndWriteOutput(i, gt_boxes, boxes, BnInOp2Blob);
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
GtBoxesAndLabels ProposalTargetKernel<T>::GetImageGtBoxes(
    size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const FloatList16* gt_boxes = BnInOp2Blob("gt_boxes")->dptr<FloatList16>(im_index);
  const Int32List16* gt_labels = BnInOp2Blob("gt_labels")->dptr<Int32List16>(im_index);
  GtBoxesAndLabels gt_boxes_and_labels(*gt_boxes, *gt_labels);

  Int32List invalid_inds;
  gt_boxes_and_labels.ForEachBox<float>([&](int32_t index, BBox<float>* box) {
    if (box->Area() <= 0) {
      invalid_inds.add_value(index);
    } else {
      FasterRcnnUtil<T>::CorrectGtBoxCoord(conf.image_height(), conf.image_width(), box);
    }
  });
  gt_boxes_and_labels.Filter(invalid_inds);

  invalid_inds.Clear();
  gt_boxes_and_labels.ForEachLabel([&](int32_t index, int32_t label) {
    if (label > conf.num_classes()) { invalid_inds.add_value(index); }
  });
  gt_boxes_and_labels.Filter(invalid_inds);

  CHECK_LE(gt_boxes_and_labels.size(), conf.max_gt_boxes_num());
  return gt_boxes_and_labels;
}

template<typename T>
typename ProposalTargetKernel<T>::BoxesWithMaxOverlap ProposalTargetKernel<T>::GetImageRoiBoxes(
    size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* rpn_rois_blob = BnInOp2Blob("rpn_rois");
  Blob* boxes_index_blob = BnInOp2Blob("boxes_index");
  auto boxes =
      GenBoxesIndex(boxes_index_blob->shape().elem_cnt(), boxes_index_blob->mut_dptr<int32_t>(),
                    rpn_rois_blob->dptr<T>(im_index), true);
  boxes.Truncate(rpn_rois_blob->shape().At(1));

  BoxesWithMaxOverlap boxes_with_max_overlap(
      boxes, BnInOp2Blob("max_overlaps")->mut_dptr<float>(),
      BnInOp2Blob("max_overlaps_gt_boxes_index")->mut_dptr<int32_t>(), true);

  return boxes_with_max_overlap;
}

template<typename T>
void ProposalTargetKernel<T>::ComputeRoiBoxesAndGtBoxesOverlaps(
    const GtBoxesAndLabels& gt_boxes, BoxesWithMaxOverlap& roi_boxes) const {
  FasterRcnnUtil<T>::ForEachOverlapBetweenBoxesAndGtBoxes(
      roi_boxes, gt_boxes, [&](int32_t index, int32_t gt_index, float overlap) {
        roi_boxes.UpdateMaxOverlap(index, gt_index, overlap);
      });
}

template<typename T>
void ProposalTargetKernel<T>::ConcatGtBoxesToRoiBoxesTail(const GtBoxesAndLabels& gt_boxes,
                                                          BoxesWithMaxOverlap& boxes) const {
  FOR_RANGE(size_t, i, 0, gt_boxes.size()) { boxes.PushBack(-(i + 1)); }
}

template<typename T>
void ProposalTargetKernel<T>::ConcatGtBoxesToRoiBoxesHead(const GtBoxesAndLabels& gt_boxes,
                                                          BoxesWithMaxOverlap& boxes) const {
  boxes.Truncate(boxes.size() + gt_boxes.size());
  for (int32_t i = boxes.size() - 1; i >= 0; --i) {
    boxes.mut_index_ptr()[i + gt_boxes.size()] = boxes.GetIndex(i);
  }
  FOR_RANGE(size_t, i, 0, gt_boxes.size()) { boxes.mut_index_ptr()[i] = (-(i + 1)); }
}

template<typename T>
void ProposalTargetKernel<T>::SubsampleForegroundAndBackground(const GtBoxesAndLabels& gt_boxes,
                                                               BoxesWithMaxOverlap& boxes) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  size_t fg_end = boxes.size();
  size_t bg_begin = boxes.size();
  size_t bg_end = boxes.size();
  boxes.SortByMaxOverlap(
      [](float lhs_overlap, float rhs_overlap) { return lhs_overlap > rhs_overlap; });
  ConcatGtBoxesToRoiBoxesHead(gt_boxes, boxes);
  boxes.ForEachMaxOverlap([&](size_t n, int32_t index, float overlap) {
    if (overlap < conf.foreground_threshold()) { fg_end = std::min(fg_end, n); }
    if (overlap < conf.background_threshold_high()) {
      bg_begin = std::min(bg_begin, n);
      boxes.set_max_overlap_gt_index(index, -1);
    }
    if (overlap < conf.background_threshold_low()) {
      bg_end = std::min(bg_end, n);
      return false;
    }
    return true;
  });
  CHECK_GE(bg_begin, fg_end);
  CHECK_GE(bg_end, bg_begin);

  size_t total_sample_cnt = conf.num_rois_per_image();
  size_t fg_cnt = total_sample_cnt * conf.foreground_fraction();
  size_t bg_cnt = 0;
  Indexes sampled_inds(total_sample_cnt, 0, boxes.mut_index_ptr(), false);
  if (fg_cnt < fg_end) {
    boxes.Shuffle(0, fg_end);
  } else {
    fg_cnt = fg_end;
  }
  sampled_inds.Concat(boxes.Slice(0, fg_cnt));
  bg_cnt = total_sample_cnt - fg_cnt;
  if (bg_cnt < bg_end - bg_begin) {
    boxes.Shuffle(bg_begin, bg_end);
  } else {
    bg_cnt = bg_end - bg_begin;
  }
  sampled_inds.Concat(boxes.Slice(bg_begin, bg_begin + bg_cnt));
  boxes.Assign(sampled_inds);
}

template<typename T>
void ProposalTargetKernel<T>::ComputeAndWriteOutput(
    size_t im_index, const GtBoxesAndLabels& gt_boxes, const BoxesWithMaxOverlap& sampled_boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  const BBoxRegressionWeights& bbox_reg_ws = conf.bbox_reg_weights();
  int64_t output_num = conf.num_rois_per_image();
  int64_t class_num = conf.num_classes();
  BBox<T>* rois_bbox = BBox<T>::MutCast(BnInOp2Blob("rois")->mut_dptr<T>(im_index));
  int32_t* labels_ptr = BnInOp2Blob("labels")->mut_dptr<int32_t>(im_index);
  int64_t bbox_im_offset = im_index * output_num;
  T* bbox_targets_ptr = BnInOp2Blob("bbox_targets")->mut_dptr<T>(bbox_im_offset);
  T* inside_weights_ptr = BnInOp2Blob("bbox_inside_weights")->mut_dptr<T>(bbox_im_offset);
  T* outside_weights_ptr = BnInOp2Blob("bbox_outside_weights")->mut_dptr<T>(bbox_im_offset);

  FOR_RANGE(size_t, i, 0, sampled_boxes.size()) {
    int32_t index = sampled_boxes.GetIndex(i);
    int32_t gt_index = sampled_boxes.GetMaxOverlapGtIndex(i);
    int32_t label = gt_boxes.GetLabel(gt_index);
    CopySampledBoxesToRois(index, sampled_boxes, gt_boxes, rois_bbox + i);
    if (label > 0) {
      labels_ptr[i] = label;
      int64_t bbox_cls_offset = i * class_num;
      auto* bbox_targets = BBoxDelta<T>::MutCast(bbox_targets_ptr) + bbox_cls_offset;
      auto* inside_weights = BBoxWeights<T>::MutCast(inside_weights_ptr) + bbox_cls_offset;
      auto* outside_weights = BBoxWeights<T>::MutCast(outside_weights_ptr) + bbox_cls_offset;
      ComputeBBoxTargets(index, gt_index, sampled_boxes, gt_boxes, bbox_reg_ws,
                         bbox_targets + label);
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
