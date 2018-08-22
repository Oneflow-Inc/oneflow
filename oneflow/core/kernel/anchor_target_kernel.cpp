#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/anchor_target_kernel.h"

namespace oneflow {

namespace {

template<typename T>
void ForEachOverlapBetweenAnchorsAndGtBoxes(
    const BBoxSlice<T>& gt_boxes_slice, const BBoxSlice<T>& anchor_boxes_slice,
    const std::function<void(int32_t, int32_t, float)>& Handler) {
  FOR_RANGE(int32_t, i, 0, gt_boxes_slice.size()) {
    FOR_RANGE(int32_t, j, 0, anchor_boxes_slice.size()) {
      float overlap = anchor_boxes_slice.GetBBox(j)->InterOverUnion(gt_boxes_slice.GetBBox(i));
      Handler(gt_boxes_slice.GetIndex(i), anchor_boxes_slice.GetIndex(j), overlap);
    }
  }
}

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

template<typename T>
void AnchorTargetKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* anchors_blob = BnInOp2Blob("anchors");
  const AnchorTargetOpConf& anchor_target_conf = op_conf().anchor_target_conf();
  const AnchorGeneratorConf& anchor_generator_conf = anchor_target_conf.anchor_generator_conf();
  float straddle_thresh = anchor_target_conf.straddle_thresh();
  FasterRcnnUtil<T>::GenerateAnchors(anchor_generator_conf, anchors_blob);
  BBoxSlice<T> anchors_slice(anchors_blob->shape().Count(0, 3), anchors_blob->dptr<T>(),
                             BnInOp2Blob("inside_anchors_index")->mut_dptr<int32_t>());
  anchors_slice.Filter([&](const BBox<T>* anchor_box) {
    return anchor_box->x1() < -straddle_thresh || anchor_box->y1() < -straddle_thresh
           || anchor_box->x2() >= anchor_generator_conf.image_width() + straddle_thresh
           || anchor_box->y2() >= anchor_generator_conf.image_height() + straddle_thresh;
  });
  *(BnInOp2Blob("inside_anchors_num")->mut_dptr<int32_t>()) = anchors_slice.size();
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BBoxSlice<T> anchor_boxes_slice = GetAnchorBoxesSlice(ctx, BnInOp2Blob);
  FOR_RANGE(size_t, image_index, 0, BnInOp2Blob("gt_boxes")->shape().At(0)) {
    BBoxSlice<T> gt_boxes_slice = GetImageGtBoxesSlice(image_index, BnInOp2Blob);
    auto labels_and_nearest_gt_boxes = ComputeOverlapsAndAssignLabels(
        image_index, gt_boxes_slice, anchor_boxes_slice, BnInOp2Blob);
    auto labeled_anchor_slice = SubsampleBackgroundsAndForegrounds(
        anchor_boxes_slice, labels_and_nearest_gt_boxes.GetLabelsPtr(),
        labels_and_nearest_gt_boxes.GetMaxOverlapsPtr());
    AssignOutputByLabels(image_index, labels_and_nearest_gt_boxes, labeled_anchor_slice,
                         BnInOp2Blob);
  }
}

template<typename T>
BBoxSlice<T> AnchorTargetKernel<T>::GetAnchorBoxesSlice(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* anchor_boxes_index_blob = BnInOp2Blob("anchor_boxes_index");
  anchor_boxes_index_blob->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("inside_anchors_index"));
  BBoxSlice<T> anchor_boxes_slice(anchors_blob->shape().Count(0, 3), anchors_blob->dptr<T>(),
                                  anchor_boxes_index_blob->mut_dptr<int32_t>(), false);
  anchor_boxes_slice.Truncate(*(BnInOp2Blob("inside_anchors_num")->dptr<int32_t>()));
  return anchor_boxes_slice;
}

template<typename T>
BBoxSlice<T> AnchorTargetKernel<T>::GetImageGtBoxesSlice(
    size_t image_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const AnchorGeneratorConf& anchor_generator_conf = conf.anchor_generator_conf();
  Blob* gt_boxes_absolute_blob = BnInOp2Blob("gt_boxes_absolute");
  int32_t boxes_num = FasterRcnnUtil<T>::ConvertGtBoxesToAbsoluteCoord(
      BnInOp2Blob("gt_boxes")->dptr<FloatList16>(image_index), anchor_generator_conf.image_height(),
      anchor_generator_conf.image_width(), gt_boxes_absolute_blob->mut_dptr<T>());
  BBoxSlice<T> gt_boxes_slice(conf.max_gt_boxes_num(), gt_boxes_absolute_blob->dptr<T>(),
                              BnInOp2Blob("gt_boxes_index")->mut_dptr<int32_t>());
  gt_boxes_slice.Truncate(boxes_num);
  return gt_boxes_slice;
}

template<typename T>
AnchorLabelsAndNearestGtBoxesInfo AnchorTargetKernel<T>::ComputeOverlapsAndAssignLabels(
    size_t image_index, const BBoxSlice<T>& gt_boxes_slice, const BBoxSlice<T>& anchor_boxes_slice,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* labels_blob = BnInOp2Blob("rpn_labels");
  Blob* anchor_max_overlaps_blob = BnInOp2Blob("anchor_max_overlaps");
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
      BnInOp2Blob("gt_boxes_nearest_anchors_index")->mut_dptr<int32_t>(),
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

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
