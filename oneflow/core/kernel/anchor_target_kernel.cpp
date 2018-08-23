#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/anchor_target_kernel.h"

namespace oneflow {
/*
namespace {

template<typename T>
void AssignPositiveLabelsToGtBoxesNearestAnchors(
    const GtBoxesNearestAnchorsInfo& gt_boxes_nearest_anchors,
    AnchorLabelsAndNearestGtBoxesInfo& anchor_labels_info) {
  gt_boxes_nearest_anchors.ForEachNearestAnchor(
      [&](int32_t anchor_idx) { anchor_labels_info.SetPositiveLabel(anchor_idx); });
}

template<typename T>
void AssignBBoxTargets(int32_t label, const BBox<T>* anchor_box, const BBox<T>* gt_box,
                       const BBoxRegressionWeights& bbox_reg_ws, BBoxDelta<T>* bbox_delta) {
  if (label == 1) { bbox_delta->TransformInverse(anchor_box, gt_box, bbox_reg_ws); }
}

template<typename T>
void AssignInsideWeights(int32_t label, BBoxWeights<T>* weights) {
  if (label == 1) {
    weights->set_weight_x(1.0);
    weights->set_weight_y(1.0);
    weights->set_weight_w(1.0);
    weights->set_weight_h(1.0);
  } else {
    weights->set_weight_x(0.0);
    weights->set_weight_y(0.0);
    weights->set_weight_w(0.0);
    weights->set_weight_h(0.0);
  }
}

template<typename T>
void AssignOutsideWeights(int32_t label, BBoxWeights<T>* weights, float reduction_coefficient) {
  if (label == 1) {
    weights->set_weight_x(reduction_coefficient);
    weights->set_weight_y(reduction_coefficient);
    weights->set_weight_w(reduction_coefficient);
    weights->set_weight_h(reduction_coefficient);
  }
}

}  // namespace
*/

template<typename T>
void AnchorTargetKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* anchors_blob = BnInOp2Blob("anchors");
  const AnchorTargetOpConf& anchor_target_conf = op_conf().anchor_target_conf();
  const AnchorGeneratorConf& anchor_generator_conf = anchor_target_conf.anchor_generator_conf();
  float straddle_thresh = anchor_target_conf.straddle_thresh();
  FasterRcnnUtil<T>::GenerateAnchors(anchor_generator_conf, anchors_blob);
  BoxesSlice<T> anchors_slice(anchors_blob->shape().Count(0, 3),
                              BnInOp2Blob("inside_anchors_index")->mut_dptr<int32_t>(),
                              anchors_blob->dptr<T>(), false);
  anchors_slice.FilterByBox([&](const BBox<T>* anchor_box) {
    return anchor_box->x1() < -straddle_thresh || anchor_box->y1() < -straddle_thresh
           || anchor_box->x2() >= anchor_generator_conf.image_width() + straddle_thresh
           || anchor_box->y2() >= anchor_generator_conf.image_height() + straddle_thresh;
  });
  *(BnInOp2Blob("inside_anchors_num")->mut_dptr<int32_t>()) = anchors_slice.size();
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto anchor_boxes_slice = GetAnchorBoxesSlice(ctx, BnInOp2Blob);
  FOR_RANGE(size_t, image_index, 0, BnInOp2Blob("gt_boxes")->shape().At(0)) {
    auto gt_boxes_slice = GetImageGtBoxesSlice(image_index, BnInOp2Blob);
    auto labels_and_nearest_gt_boxes =
        ComputeOverlapsAndSetLabels(image_index, gt_boxes_slice, anchor_boxes_slice, BnInOp2Blob);
    size_t fg_cnt = SubsampleForeground(labels_and_nearest_gt_boxes);
    size_t bg_cnt = SubsampleBackground(fg_cnt, labels_and_nearest_gt_boxes);
    WriteOutput(image_index, fg_cnt + bg_cnt, labels_and_nearest_gt_boxes, BnInOp2Blob);
  }
}

template<typename T>
BoxesSlice<T> AnchorTargetKernel<T>::GetAnchorBoxesSlice(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* anchor_boxes_index_blob = BnInOp2Blob("anchor_boxes_index");
  anchor_boxes_index_blob->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("inside_anchors_index"));
  BoxesSlice<T> anchor_boxes_slice(anchors_blob->shape().Count(0, 3),
                                   anchor_boxes_index_blob->mut_dptr<int32_t>(),
                                   anchors_blob->dptr<T>(), false);
  anchor_boxes_slice.Truncate(*(BnInOp2Blob("inside_anchors_num")->dptr<int32_t>()));
  return anchor_boxes_slice;
}

template<typename T>
BoxesSlice<T> AnchorTargetKernel<T>::GetImageGtBoxesSlice(
    size_t image_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const AnchorGeneratorConf& anchor_generator_conf = conf.anchor_generator_conf();
  Blob* gt_boxes_absolute_blob = BnInOp2Blob("gt_boxes_absolute");
  int32_t boxes_num = FasterRcnnUtil<T>::ConvertGtBoxesToAbsoluteCoord(
      BnInOp2Blob("gt_boxes")->dptr<FloatList16>(image_index), anchor_generator_conf.image_height(),
      anchor_generator_conf.image_width(), gt_boxes_absolute_blob->mut_dptr<T>());
  BoxesSlice<T> gt_boxes_slice(conf.max_gt_boxes_num(),
                               BnInOp2Blob("gt_boxes_index")->mut_dptr<int32_t>(),
                               gt_boxes_absolute_blob->dptr<T>());
  gt_boxes_slice.Truncate(boxes_num);
  return gt_boxes_slice;
}

template<typename T>
typename AnchorTargetKernel<T>::BoxesLabelsAndNearestGtBoxes
AnchorTargetKernel<T>::ComputeOverlapsAndSetLabels(
    size_t image_index, const BoxesSlice<T>& gt_boxes_slice,
    const BoxesSlice<T>& anchor_boxes_slice,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const float positive_overlap_threshold =
      op_conf().anchor_target_conf().positive_overlap_threshold();
  Blob* max_overlaps_blob = BnInOp2Blob("max_overlaps");
  Blob* gt_max_overlaps_blob = BnInOp2Blob("gt_max_overlaps");
  std::memset(max_overlaps_blob->mut_dptr(), 0,
              max_overlaps_blob->shape().elem_cnt() * sizeof(float));
  std::memset(gt_max_overlaps_blob->mut_dptr(), 0,
              gt_max_overlaps_blob->shape().elem_cnt() * sizeof(float));

  auto boxes_to_nearest_gt_boxes = GenBoxesToNearestGtBoxesSlice(
      anchor_boxes_slice, max_overlaps_blob->mut_dptr<float>(),
      BnInOp2Blob("anchor_nearest_gt_box_index")->mut_dptr<int32_t>());
  auto gt_boxes_to_nearest_boxes = GenGtBoxesToNearestBoxesSlice(
      gt_boxes_slice, gt_max_overlaps_blob->mut_dptr<float>(),
      BnInOp2Blob("gt_box_nearest_anchor_index")->mut_dptr<int32_t>());
  auto boxes_labels = GenLabeledBoxesSlice<3>(
      boxes_to_nearest_gt_boxes, BnInOp2Blob("rpn_labels")->mut_dptr<int32_t>(image_index));

  FasterRcnnUtil<T>::ForEachOverlapBetweenBoxesAndGtBoxes(
      anchor_boxes_slice, gt_boxes_slice,
      [&](int32_t box_index, int32_t gt_box_index, float overlap) {
        boxes_to_nearest_gt_boxes.UpdateMaxOverlapGtBox(box_index, gt_box_index, overlap, [&]() {
          if (overlap >= positive_overlap_threshold) {
            boxes_labels.set_label(box_index, 1);
          } else {
            boxes_labels.set_label(box_index, -1);
          }
        });
        gt_boxes_to_nearest_boxes.UpdateNearestBox(gt_box_index, box_index, overlap);
      });

  gt_boxes_to_nearest_boxes.ForEachNearestBox(
      [&](int32_t box_index) { boxes_labels.set_label(box_index, 1); });

  return boxes_labels;
}

template<typename T>
size_t AnchorTargetKernel<T>::SubsampleForeground(
    BoxesLabelsAndNearestGtBoxes& boxes_labels_and_nearest_gt_boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t fg_cnt = conf.batch_size_per_image() * conf.foreground_fraction();
  boxes_labels_and_nearest_gt_boxes.GroupByLabel();
  return boxes_labels_and_nearest_gt_boxes.Subsample(1, fg_cnt, [](int32_t) {},
                                                     [&](int32_t disable_box_index) {
                                                       boxes_labels_and_nearest_gt_boxes.set_label(
                                                           disable_box_index, -1);
                                                     });
}

template<typename T>
size_t AnchorTargetKernel<T>::SubsampleBackground(
    size_t fg_cnt, BoxesLabelsAndNearestGtBoxes& boxes_labels_and_nearest_gt_boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t bg_cnt = conf.batch_size_per_image() - fg_cnt;
  boxes_labels_and_nearest_gt_boxes.SortByOverlap(
      [](float lhs_overlap, float rhs_overlap) { return lhs_overlap < rhs_overlap; });
  int32_t negative_end = boxes_labels_and_nearest_gt_boxes.FindByOverlap(
      [&](float overlap) { return overlap >= conf.negative_overlap_threshold(); });
  size_t actual_bg_cnt =
      (negative_end >= 0) ? negative_end : boxes_labels_and_nearest_gt_boxes.size();
  if (actual_bg_cnt > bg_cnt) {
    boxes_labels_and_nearest_gt_boxes.Shuffle(0, actual_bg_cnt);
    actual_bg_cnt = bg_cnt;
  }
  FOR_RANGE(size_t, i, 0, actual_bg_cnt) {
    int32_t box_index = boxes_labels_and_nearest_gt_boxes.GetIndex(i);
    boxes_labels_and_nearest_gt_boxes.set_label(box_index, 0);
  }
  return actual_bg_cnt;
}

template<typename T>
void AnchorTargetKernel<T>::WriteOutput(
    size_t image_index, size_t total_sample_count,
    const BoxesLabelsAndNearestGtBoxes& labels_and_nearest_gt_boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  CHECK_GT(total_sample_count, 0);
  const BBox<T>* gt_boxes = BBox<T>::Cast(BnInOp2Blob("gt_boxes_absolute")->dptr<T>());
  BBoxDelta<T>* bbox_target =
      BBoxDelta<T>::MutCast(BnInOp2Blob("rpn_bbox_targets")->mut_dptr<T>(image_index));
  BBoxWeights<T>* inside_weights =
      BBoxWeights<T>::MutCast(BnInOp2Blob("rpn_bbox_inside_weights")->mut_dptr<T>(image_index));
  BBoxWeights<T>* outside_weights =
      BBoxWeights<T>::MutCast(BnInOp2Blob("rpn_bbox_outside_weights")->mut_dptr<T>(image_index));
  const BBoxRegressionWeights& bbox_reg_ws = op_conf().anchor_target_conf().bbox_reg_weights();
  const float reduction_coefficient = 1 / total_sample_count;

  FOR_RANGE(size_t, i, 0, BnInOp2Blob("anchors")->shape().Count(0, 3)) {
    const BBox<T>* anchor_box = labels_and_nearest_gt_boxes.bbox(i);
    const BBox<T>* gt_box = gt_boxes + labels_and_nearest_gt_boxes.max_overlap_gt_box_index(i);
    int32_t label = labels_and_nearest_gt_boxes.label(i);

    if (label == 1) {
      bbox_target->TransformInverse(anchor_box, gt_box, bbox_reg_ws);
      inside_weights->set_weight_x(1.0);
      inside_weights->set_weight_y(1.0);
      inside_weights->set_weight_w(1.0);
      inside_weights->set_weight_h(1.0);
      outside_weights->set_weight_x(reduction_coefficient);
      outside_weights->set_weight_y(reduction_coefficient);
      outside_weights->set_weight_w(reduction_coefficient);
      outside_weights->set_weight_h(reduction_coefficient);
    } else {
      inside_weights->set_weight_x(0.0);
      inside_weights->set_weight_y(0.0);
      inside_weights->set_weight_w(0.0);
      inside_weights->set_weight_h(0.0);
    }
  }
}

/*
template<typename T>
AnchorLabelsAndNearestGtBoxesInfo AnchorTargetKernel<T>::ComputeOverlapsAndAssignLabels(
    size_t image_index, const BBoxSlice<T>& gt_boxes_slice, const BBoxSlice<T>& anchor_boxes_slice,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* labels_blob = BnInOp2Blob("rpn_labels");
  Blob* anchor_max_overlaps_blob = BnInOp2Blob("max_overlaps");
  Blob* gt_max_overlaps_blob = BnInOp2Blob("gt_max_overlaps");
  std::memset(anchor_max_overlaps_blob->mut_dptr(), 0,
              anchor_max_overlaps_blob->shape().elem_cnt() * sizeof(float));
  std::memset(gt_max_overlaps_blob->mut_dptr(), 0,
              gt_max_overlaps_blob->shape().elem_cnt() * sizeof(float));

  AnchorLabelsAndNearestGtBoxesInfo anchor_labels_info(
      labels_blob->mut_dptr<int32_t>(image_index), anchor_max_overlaps_blob->mut_dptr<float>(),
      BnInOp2Blob("anchor_nearest_gt_box_index")->mut_dptr<int32_t>(),
      op_conf().anchor_target_conf().positive_overlap_threshold(),
      op_conf().anchor_target_conf().negative_overlap_threshold(), labels_blob->shape().Count(1));
  GtBoxesNearestAnchorsInfo gt_boxes_nearest_anchors(
      BnInOp2Blob("gt_box_nearest_anchor_index")->mut_dptr<int32_t>(),
      gt_max_overlaps_blob->mut_dptr<float>());
  ForEachOverlapBetweenAnchorsAndGtBoxes(
      gt_boxes_slice, anchor_boxes_slice,
      [&](int32_t gt_box_idx, int32_t anchor_box_idx, float overlap) {
        anchor_labels_info.AssignLabelByOverlapThreshold(anchor_box_idx, gt_box_idx, overlap);
        gt_boxes_nearest_anchors.TryRecordAnchorAsNearest(gt_box_idx, anchor_box_idx, overlap);
      });
  AssignPositiveLabelsToGtBoxesNearestAnchors<T>(gt_boxes_nearest_anchors, anchor_labels_info);

  return anchor_labels_info;
}

template<typename T>
LabeledBBoxSlice<T, 3> AnchorTargetKernel<T>::SubsampleBackgroundsAndForegrounds(
    BBoxSlice<T>& anchor_boxes_slice, int32_t* anchor_labels_ptr,
    const float* max_overlaps_ptr) const {
  LabeledBBoxSlice<T, 3> labeled_anchor_slice(anchor_boxes_slice, anchor_labels_ptr);
  size_t batch_size_per_image = op_conf().anchor_target_conf().batch_size_per_image();
  float foreground_fraction = op_conf().anchor_target_conf().foreground_fraction();
  // subsample foregrounds by labels
  labeled_anchor_slice.GroupByLabel();
  size_t fg_cnt = batch_size_per_image * foreground_fraction;
  fg_cnt = labeled_anchor_slice.SubsampleByLabel(1, fg_cnt);
  // subsample backgrounds by overlaps
  float negative_threshold = op_conf().anchor_target_conf().negative_overlap_threshold();
  size_t bg_cnt = batch_size_per_image - fg_cnt;
  labeled_anchor_slice.SubsampleByOverlap(max_overlaps_ptr, negative_threshold, bg_cnt);

  return labeled_anchor_slice;
}

template<typename T>
void AnchorTargetKernel<T>::AssignOutputByLabels(
    size_t image_index, const AnchorLabelsAndNearestGtBoxesInfo& labels_and_nearest_gt_boxes,
    const LabeledBBoxSlice<T, 3>& labeled_anchor_slice,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  const BBox<T>* anchor_boxes = BBox<T>::Cast(anchors_blob->dptr<T>());
  const BBox<T>* gt_boxes = BBox<T>::Cast(BnInOp2Blob("gt_boxes_absolute")->dptr<T>());
  BBoxDelta<T>* bbox_target =
      BBoxDelta<T>::MutCast(BnInOp2Blob("rpn_bbox_targets")->mut_dptr<T>(image_index));
  const int32_t* anchor_labels_ptr = labels_and_nearest_gt_boxes.GetLabelsPtr();
  BBoxWeights<T>* inside_weights =
      BBoxWeights<T>::MutCast(BnInOp2Blob("rpn_bbox_inside_weights")->mut_dptr<T>(image_index));
  BBoxWeights<T>* outside_weights =
      BBoxWeights<T>::MutCast(BnInOp2Blob("rpn_bbox_outside_weights")->mut_dptr<T>(image_index));
  const BBoxRegressionWeights& bbox_reg_ws = op_conf().anchor_target_conf().bbox_reg_weights();
  const size_t fg_cnt = labeled_anchor_slice.GetLabelCount(1);
  const size_t bg_cnt = labeled_anchor_slice.GetLabelCount(0);
  CHECK_GT(fg_cnt, 0);
  CHECK_GT(bg_cnt, 0);
  const float reduction_coefficient = 1 / (fg_cnt + bg_cnt);

  int64_t anchors_num = anchors_blob->shape().Count(0, 3);
  FOR_RANGE(size_t, i, 0, anchors_num) {
    int32_t label = anchor_labels_ptr[i];
    int32_t nearest_gt_box_index = labels_and_nearest_gt_boxes.GetNearestGtBoxesPtr()[i];
    AssignBBoxTargets(label, &anchor_boxes[i], &gt_boxes[nearest_gt_box_index], bbox_reg_ws,
                      &bbox_target[i]);
    AssignInsideWeights(label, &inside_weights[i]);
    AssignOutsideWeights(label, &outside_weights[i], reduction_coefficient);
  }
}
*/

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
