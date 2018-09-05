#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/anchor_target_kernel.h"

namespace oneflow {

template<typename T>
void AnchorTargetKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* anchors_blob = BnInOp2Blob("anchors");
  const AnchorTargetOpConf& anchor_target_conf = op_conf().anchor_target_conf();
  const AnchorGeneratorConf& anchor_generator_conf = anchor_target_conf.anchor_generator_conf();
  float straddle_thresh = anchor_target_conf.straddle_thresh();
  FasterRcnnUtil<T>::GenerateAnchors(anchor_generator_conf, anchors_blob);
  auto inside_inds = GenBoxesIndex(anchors_blob->shape().Count(0, 3),
                                   BnInOp2Blob("inside_anchors_index")->mut_dptr<int32_t>(),
                                   anchors_blob->dptr<T>(), true);
  inside_inds.FilterByBBox([&](size_t n, int32_t index, const BBox<T>* bbox) {
    return bbox->x1() < -straddle_thresh || bbox->y1() < -straddle_thresh
           || bbox->x2() >= anchor_generator_conf.image_width() + straddle_thresh
           || bbox->y2() >= anchor_generator_conf.image_height() + straddle_thresh;
  });
  *(BnInOp2Blob("inside_anchors_num")->mut_dptr<int32_t>()) = inside_inds.size();
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(size_t, im_index, 0, BnInOp2Blob("gt_boxes")->shape().At(0)) {
    auto gt_boxes = GetImageGtBoxes(im_index, BnInOp2Blob);
    auto anchor_boxes = GetImageAnchorBoxes(ctx, im_index, BnInOp2Blob);
    ComputeOverlapsAndSetLabels(gt_boxes, anchor_boxes);
    size_t fg_cnt = SubsampleForeground(anchor_boxes);
    size_t bg_cnt = SubsampleBackground(fg_cnt, anchor_boxes);
    // size_t fg_cnt = ChoiceForeground(anchor_boxes);
    // size_t bg_cnt = ChoiceBackground(fg_cnt, anchor_boxes);
    ComputeTargetsAndWriteOutput(im_index, fg_cnt + bg_cnt, gt_boxes, anchor_boxes, BnInOp2Blob);
  }
}

template<typename T>
typename AnchorTargetKernel<T>::BoxesLabelAndMaxOverlap AnchorTargetKernel<T>::GetImageAnchorBoxes(
    const KernelCtx& ctx, size_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* anchor_boxes_index_blob = BnInOp2Blob("anchor_boxes_index");
  anchor_boxes_index_blob->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("inside_anchors_index"));
  auto anchor_boxes =
      GenBoxesIndex(anchors_blob->shape().Count(0, 3), anchor_boxes_index_blob->mut_dptr<int32_t>(),
                    anchors_blob->dptr<T>(), false);
  anchor_boxes.Truncate(*(BnInOp2Blob("inside_anchors_num")->dptr<int32_t>()));

  BoxesWithMaxOverlap boxes_with_max_overlap(
      anchor_boxes, BnInOp2Blob("max_overlaps")->mut_dptr<float>(),
      BnInOp2Blob("anchor_nearest_gt_box_index")->mut_dptr<int32_t>(), true);

  Blob* rpn_labels_blob = BnInOp2Blob("rpn_labels");
  BoxesLabelAndMaxOverlap labeled_boxes_with_max_overlap(
      boxes_with_max_overlap, rpn_labels_blob->mut_dptr<int32_t>(im_index));
  labeled_boxes_with_max_overlap.FillLabel(0, rpn_labels_blob->shape().Count(1), -1);

  return labeled_boxes_with_max_overlap;
}

template<typename T>
GtBoxesWithMaxOverlap AnchorTargetKernel<T>::GetImageGtBoxes(
    size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const AnchorGeneratorConf& anchor_generator_conf =
      op_conf().anchor_target_conf().anchor_generator_conf();
  GtBoxesWithMaxOverlap gt_boxes(*(BnInOp2Blob("gt_boxes")->dptr<FloatList16>(im_index)));
  Int32List invalid_gt_inds;
  gt_boxes.ForEachBox<float>([&](int32_t index, BBox<float>* box) {
    if (box->Area() <= 0) {
      invalid_gt_inds.add_value(index);
    } else {
      FasterRcnnUtil<T>::CorrectGtBoxCoord(anchor_generator_conf.image_height(),
                                           anchor_generator_conf.image_width(), box);
    }
  });
  gt_boxes.Filter(invalid_gt_inds);

  return gt_boxes;
}

template<typename T>
void AnchorTargetKernel<T>::ComputeOverlapsAndSetLabels(
    GtBoxesWithMaxOverlap& gt_boxes, BoxesLabelAndMaxOverlap& anchor_boxes) const {
  float positive_overlap_threshold = op_conf().anchor_target_conf().positive_overlap_threshold();
  FasterRcnnUtil<T>::ForEachOverlapBetweenBoxesAndGtBoxes(
      anchor_boxes, gt_boxes, [&](int32_t index, int32_t gt_index, float overlap) {
        anchor_boxes.UpdateMaxOverlap(index, gt_index, overlap, [&]() {
          if (overlap >= positive_overlap_threshold) { anchor_boxes.set_label(index, 1); }
        });
        gt_boxes.UpdateMaxOverlap(gt_index, index, overlap);
      });

  gt_boxes.ForEachMaxOverlapWithIndex([&](int32_t index) { anchor_boxes.set_label(index, 1); });
}

template<typename T>
size_t AnchorTargetKernel<T>::SubsampleForeground(BoxesLabelAndMaxOverlap& boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t fg_cnt = conf.batch_size_per_image() * conf.foreground_fraction();
  boxes.SortByLabel([](int32_t lhs_label, int32_t rhs_label) { return lhs_label > rhs_label; });
  size_t fg_end = boxes.FindByLabel([](int32_t label) { return label != 1; });
  if (fg_end > fg_cnt) {
    boxes.Shuffle(0, fg_end);
    FOR_RANGE(size_t, i, fg_cnt, fg_end) { boxes.SetLabel(i, -1); }
  } else {
    fg_cnt = fg_end;
  }
  return fg_cnt;
}

template<typename T>
size_t AnchorTargetKernel<T>::SubsampleBackground(size_t fg_cnt,
                                                  BoxesLabelAndMaxOverlap& boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t bg_cnt = conf.batch_size_per_image() - fg_cnt;
  boxes.SortByMaxOverlap(
      [](float lhs_overlap, float rhs_overlap) { return lhs_overlap < rhs_overlap; });
  size_t bg_end = boxes.FindByMaxOverlap(
      [&](float overlap) { return overlap >= conf.negative_overlap_threshold(); });
  if (bg_end > bg_cnt) {
    boxes.Shuffle(0, bg_end);
  } else {
    bg_cnt = bg_end;
  }
  FOR_RANGE(size_t, i, 0, bg_cnt) { boxes.SetLabel(i, 0); }
  return bg_cnt;
}

template<typename T>
void AnchorTargetKernel<T>::ComputeTargetsAndWriteOutput(
    size_t im_index, size_t total_sample_count, const GtBoxesWithMaxOverlap& gt_boxes,
    const BoxesLabelAndMaxOverlap& anchor_boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  CHECK_GT(total_sample_count, 0);
  const float reduction_coefficient = 1.f / total_sample_count;
  BBoxDelta<T>* bbox_target =
      BBoxDelta<T>::MutCast(BnInOp2Blob("rpn_bbox_targets")->mut_dptr<T>(im_index));
  BBoxWeights<T>* inside_weights =
      BBoxWeights<T>::MutCast(BnInOp2Blob("rpn_bbox_inside_weights")->mut_dptr<T>(im_index));
  BBoxWeights<T>* outside_weights =
      BBoxWeights<T>::MutCast(BnInOp2Blob("rpn_bbox_outside_weights")->mut_dptr<T>(im_index));
  const BBoxRegressionWeights& bbox_reg_ws = op_conf().anchor_target_conf().bbox_reg_weights();

  FOR_RANGE(int32_t, i, 0, anchor_boxes.capacity()) {
    int32_t label = anchor_boxes.label(i);
    if (label == 1) {
      const BBox<T>* anchor_box = anchor_boxes.bbox(i);
      const BBox<float>* gt_box = gt_boxes.GetBBox<float>(anchor_boxes.max_overlap_gt_index(i));
      bbox_target[i].TransformInverse(anchor_box, gt_box, bbox_reg_ws);
      inside_weights[i].set_weight_x(1.0);
      inside_weights[i].set_weight_y(1.0);
      inside_weights[i].set_weight_w(1.0);
      inside_weights[i].set_weight_h(1.0);
      outside_weights[i].set_weight_x(reduction_coefficient);
      outside_weights[i].set_weight_y(reduction_coefficient);
      outside_weights[i].set_weight_w(reduction_coefficient);
      outside_weights[i].set_weight_h(reduction_coefficient);
    } else {
      inside_weights[i].set_weight_x(0.0);
      inside_weights[i].set_weight_y(0.0);
      inside_weights[i].set_weight_w(0.0);
      inside_weights[i].set_weight_h(0.0);
    }
  }
}

template<typename T>
size_t AnchorTargetKernel<T>::ChoiceForeground(BoxesLabelAndMaxOverlap& boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t fg_cnt = conf.batch_size_per_image() * conf.foreground_fraction();
  size_t fg_num = 0;
  boxes.ForEachLabel([&](size_t n, int32_t index, int32_t label) {
    if (label == 1) {
      if (fg_num >= fg_cnt) { boxes.set_label(index, -1); }
      ++fg_num;
    }
    return true;
  });
  if (fg_num < fg_cnt) { fg_cnt = fg_num; }
  return fg_cnt;
}
template<typename T>
size_t AnchorTargetKernel<T>::ChoiceBackground(size_t fg_cnt,
                                               BoxesLabelAndMaxOverlap& boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t bg_cnt = conf.batch_size_per_image() - fg_cnt;
  size_t bg_num = 0;
  boxes.ForEachMaxOverlap([&](size_t n, int32_t index, float overlap) {
    if (overlap < conf.negative_overlap_threshold()) {
      if (bg_num < bg_cnt) { boxes.set_label(index, 0); }
      ++bg_num;
    }
    return true;
  });
  if (bg_num < bg_cnt) { bg_cnt = bg_num; }
  return bg_cnt;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
