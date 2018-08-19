#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/anchor_target_kernel.h"

namespace oneflow {

namespace {

template<typename T>
void ForEachOverlapBetweenInsideAnchorsAndGtBoxes(
    const BBoxSlice<T>& gt_boxes_slice, const BBoxSlice<T>& anchor_boxes_slice,
    const std::function<void(int32_t, int32_t, float)>& Handler) {
  FOR_RANGE(int32_t, i, 0, gt_boxes_slice.size()) {
    FOR_RANGE(int32_t, j, 0, anchor_boxes_slice.size()) {
      float overlap = anchor_boxes_slice.GetBBox(j)->InterOverUnion(gt_boxes_slice.GetBBox(i));
      Handler(gt_boxes_slice.GetSlice(i), anchor_boxes_slice.GetSlice(j), overlap);
    }
  }
}

void AssignPositiveLabelsToGtBoxesNearestAnchors(
    const GtBoxesNearestAnchorsInfo& gt_boxes_nearest_anchors,
    AnchorLabelsAndMaxOverlapsInfo& anchor_labels_info) {
  gt_boxes_nearest_anchors.ForEachNearestAnchor(
      [&](int32_t anchor_idx) { anchor_labels_info.TrySetPositiveLabel(anchor_idx); });
}

template<typename T>
void SetValue(int32_t size, T* data_ptr, T value) {
  FOR_RANGE(int32_t, i, 0, size) { data_ptr[i] = value; }
}

}  // namespace

template<typename T>
void AnchorTargetKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  const AnchorGeneratorConf& anchor_generator_conf = GetCustomizedOpConf().anchors_generator_conf();
  FasterRcnnUtil<T>::GenerateAnchors(anchor_generator_conf, anchors_blob);
  BBoxSlice<T> anchors_slice(anchors_blob->shape().elem_cnt(), anchors_blob->dptr<T>(),
                             BnInOp2Blob("anchors_index")->mut_dptr<T>());
  anchors_slice.Filter([&](const BBox<T>* anchor_box) {
    return anchor_box->x1() < 0 || anchor_box->y1() < 0
           || anchor_box->x2() >= anchor_generator_conf.image_width
           || anchor_box->y2() >= anchor_generator_conf.image_height;
  });
  *(BnInOp2Blob("inside_anchor_num")->mut_dptr<int32_t>()) = inside_anchors_slice.size();
}

template<typename T>
AnchorLabelsAndMaxOverlapsInfo AnchorTargetKernel<T>::AssignLabels(
    const BBoxSlice<T>& gt_boxes_slice, const BBoxSlice<T>& anchor_boxes_slice,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  // From anchor perspective
  // "anchor_label" (H, W, A)                           label
  // "anchor_max_overlaps" (H, W, A)                    overlap
  // "anchor_max_overlap_gt_boxes_index" (H, W, A)      gt_box_index
  Blob* anchor_labels_blob = BnInOp2Blob("anchor_labels");
  AnchorLabelsAndMaxOverlapsInfo anchor_labels_info(
      anchor_labels_blob->mut_dptr<T>(), BnInOp2Blob("anchor_max_overlaps")->mut_dptr<T>(),
      BnInOp2Blob("anchor_max_overlap_gt_boxes_index")->mut_dptr<T>(),
      GetCustomizedOpConf().positive_overlap_threshold(),
      GetCustomizedOpConf().negative_overlap_threshold(), anchor_labels_blob->shape().elem_cnt());
  // From gt_box perspective
  // "gt_boxes_nearest_anchors_index" (max_gt_boxes_num * H * W * A)
  // "gt_max_overlaps" (max_gt_boxes_num, 1)
  GtBoxesNearestAnchorsInfo gt_boxes_nearest_anchors(
      BnInOp2Blob("gt_boxes_nearest_anchors_index")->mut_dptr<T>(),
      BnInOp2Blob("gt_max_overlaps")->mut_dptr<T>());

  ForEachOverlapBetweenInsideAnchorsAndGtBoxes(
      gt_boxes_slice, anchor_boxes_slice,
      [&](int32_t gt_box_idx, int32_t anchor_box_idx, float overlap) {
        anchor_labels_info.AssignLabelByOverlapThreshold(anchor_box_idx, gt_box_idx, overlap);
        gt_boxes_nearest_anchors.TryRecordAnchorAsNearest(gt_box_idx, anchor_box_idx, overlap, );
      });
  AssignPositiveLabelsToGtBoxesNearestAnchors(gt_boxes_nearest_anchors, anchor_labels_info);

  return anchor_labels_info;
}

template<typename T>
void AnchorTargetKernel<T>::SubsamplePositiveAndNegativeLabels(LabelBBoxSlice& labeled_anchor_slice,
                                                               size_t image_index) {
  labeled_anchor_slice.GroupByLabel();
  size_t batch_size_per_image = GetCustomizedOpConf().batch_size_per_image();
  float foreground_fraction = GetCustomizedOpConf().foreground_fraction();
  size_t fg_cnt = batch_size_per_image * fg_ratio;
  fg_cnt = labeled_anchor_slice.Subsample(1, fg_cnt);
  size_t bg_cnt = batch_size_per_image - fg_cnt;
  labeled_anchor_slice.Subsample(0, bg_cnt);
}

template<typename T>
void AnchorTargetKernel<T>::WriteToOutputBlobs(
    const LabeledBBoxSlice<T>& labeled_anchor_slice, const BBoxSlice<T>& anchor_boxes_slice,
    const AnchorLabelsAndMaxOverlapsInfo& anchor_label_and_nearest_gt_box,
    const BBoxSlice<T>& gt_boxes_slice, int32_t* rpn_labels_ptr, T* rpn_bbox_targets_ptr,
    T* rpn_bbox_inside_weights_ptr, T* rpn_bbox_outside_weights_ptr) {
  // 初始化“rpn_labels”, "rpn_bbox_inside_weights", "rpn_bbox_outside_weights"
  // 不用初始化"rpn_bbox_targets"，结果的正确性可由inside_weights和outside_weights保证
  size_t anchors_num = anchor_boxes_slice.capacity();
  SetValue(anchors_num, rpn_labels_ptr, -1);
  Memset<DeviceType::kCPU>(ctx.device_ctx, rpn_bbox_inside_weights_blob->mut_dptr(), 0,
                           rpn_bbox_inside_weights_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx.device_ctx, rpn_bbox_outside_weights_blob->mut_dptr(), 0,
                           rpn_bbox_outside_weights_blob->ByteSizeOfDataContentField());

  const BBoxRegressionWeights& bbox_reg_ws = GetCustomizedOpConf().bbox_reg_weights();
  BBox<T>* rpn_bbox_inside_weights_ptr =
      BBox<T>::MutCast(rpn_bbox_inside_weights_ptr->mut_dptr<T>());
  BBox<T>* rpn_bbox_outside_weights_ptr =
      BBox<T>::MutCast(rpn_bbox_outside_weights_ptr->mut_dptr<T>());
  const size_t fg_cnt = labeled_anchor_slice.GetLabelCount(1);
  const size_t bg_cnt = labeled_anchor_slice.GetLabelCount(0);
  const float reduction_coefficient = 1 / (fg_cnt + bg_cnt);
  FOR_RANGE(size_t, i, 0, labeled_anchor_slice->size()) {
    size_t anchor_index = labeled_anchor_slice->GetIndex(i);
    int32_t label = labeled_anchor_slice->GetLabel(i);
    rpn_labels_ptr[anchor_index] = label;
    switch (label) {
      case 1:
        // rpn_bbox_targets
        size_t gt_box_index = anchor_label_and_nearest_gt_box->GetNearstGtBoxes()[i];
        BBox<T>* gt_box = gt_boxes_slice->GetBBox(gt_box_index);
        BBox<T>* anchor_box = anchor_boxes_slice->GetBBox(anchor_index);
        BBoxDelta<T>* bbox_target = BBoxDelta::Cast(rpn_bbox_targets_ptr + anchor_index);
        bbox_target->TransformInverse(anchor_box, gt_box, bbox_reg_ws);
        // rpn_bbox_inside_weights
        rpn_bbox_inside_weights_ptr[anchor_index].set_x1(1.0);
        rpn_bbox_inside_weights_ptr[anchor_index].set_y1(1.0);
        rpn_bbox_inside_weights_ptr[anchor_index].set_x2(1.0);
        rpn_bbox_inside_weights_ptr[anchor_index].set_y2(1.0);
        // rpn_bbox_outside_weights
        rpn_bbox_outside_weights_ptr[anchor_index].set_x1(reduction_coefficient);
        rpn_bbox_outside_weights_ptr[anchor_index].set_y1(reduction_coefficient);
        rpn_bbox_outside_weights_ptr[anchor_index].set_x2(reduction_coefficient);
        rpn_bbox_outside_weights_ptr[anchor_index].set_y2(reduction_coefficient);
        break;
      case 0:
        // rpn_bbox_inside_weights
        rpn_bbox_inside_weights_ptr[anchor_index].set_x1(0.0);
        rpn_bbox_inside_weights_ptr[anchor_index].set_y1(0.0);
        rpn_bbox_inside_weights_ptr[anchor_index].set_x2(0.0);
        rpn_bbox_inside_weights_ptr[anchor_index].set_y2(0.0);
        break;
      default:
    }
  }
}

template<typename T>
const PbMessage& AnchorTargetKernel<T>::GetCustomizedOpConf() const {
  return this->op_conf().anchors_generator_conf();
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* anchor_boxes_index_blob = BnInOp2Blob("anchor_boxes_index_blob");
  anchor_boxes_index_blob->CopyFrom(BnInOp2Blob("anchors_index"));
  Blob* rpn_labels = BnInOp2Blob("rpn_labels");
  Blob* rpn_bbox_targets = BnInOp2Blob("rpn_bbox_targets");
  Blob* rpn_bbox_inside_weights = BnInOp2Blob("rpn_bbox_inside_weights");
  Blob* rpn_bbox_outside_weights = BnInOp2Blob("rpn_bbox_outside_weights");

  Blob* gt_boxes_absolute_blob = BnInOp2Blob("gt_boxes_absolute");
  BBoxSlice<T> anchor_boxes_slice(anchors_blob->shape().elem_cnt(), anchors_blob->dptr<T>(),
                                  anchor_boxes_index_blob->mut_dptr<T>(), false);
  anchor_boxes_slice.Truncate(*(BnInOp2Blob("inside_anchor_num")->dptr<int32_t>()));
  FOR_RANGE(int64_t, image_index, 0, images_num) {
    int32_t boxes_num = FasterRcnnUtil<T>::ConvertGtBoxesToAbsoluteCoord(
        gt_boxes_blob->dptr<FloatList16>(image_index), gt_boxes_absolute_blob->mut_dptr<T>());
    BBoxSlice<T> gt_boxes_slice(GetCustomizedOpConf().max_gt_boxes_num(),
                                gt_boxes_absolute_blob->dptr<T>(),
                                BnInOp2Blob("gt_boxes_index")->mut_dptr<T>());
    gt_boxes_slice.Truncate(boxes_num);
    AnchorLabelsAndMaxOverlapsInfo anchor_label_and_nearest_gt_box =
        AssignLabels(gt_boxes_slice, anchor_boxes_slice, BnInOp2Blob);
    LabeledBBoxSlice<size_t, 3> labeled_anchor_slice(
        anchor_boxes_slice, anchor_label_and_nearest_gt_box.GetAnchorLabels());
  }
  WriteToOutputBlobs(labeled_anchor_slice, anchor_boxes_slice, gt_boxes_slice,
                     BnInOp2Blob("rpn_labels")->mut_dptr<int32_t>(image_index),
                     BnInOp2Blob("rpn_bbox_targets")->mut_dptr<T>(image_index),
                     BnInOp2Blob("rpn_bbox_inside_weights")->mut_dptr<T>(image_index),
                     BnInOp2Blob("rpn_bbox_outside_weights")->mut_dptr<T>(image_index) s);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
