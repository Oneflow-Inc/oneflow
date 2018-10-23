#include "oneflow/core/kernel/anchor_target_kernel.h"

namespace oneflow {

namespace {

template<typename BBox, typename GtBBox>
void ForEachOverlapBetweenBoxesAndGtBoxes(
    const BBoxIndices<IndexSequence, BBox>& boxes,
    const BBoxIndices<IndexSequence, GtBBox>& gt_boxes,
    const std::function<void(int32_t, int32_t, float)>& Handler) {
  FOR_RANGE(size_t, i, 0, gt_boxes.size()) {
    FOR_RANGE(size_t, j, 0, boxes.size()) {
      float overlap = boxes.GetBBox(j)->InterOverUnion(gt_boxes.GetBBox(i));
      Handler(boxes.GetIndex(j), gt_boxes.GetIndex(i), overlap);
    }
  }
}

}  // namespace

template<typename T>
void AnchorTargetKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const float straddle_thresh = conf.straddle_thresh();
  int32_t im_height = -1;
  int32_t im_width = -1;
  Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* inside_inds_blob = BnInOp2Blob("anchors_inside_inds");
  size_t num_anchors = 0;
  for (const AnchorGeneratorConf& anchor_generator_conf : conf.anchor_generator_conf()) {
    if (im_height == -1) {
      im_height = anchor_generator_conf.image_height();
      im_width = anchor_generator_conf.image_width();
    } else {
      CHECK_EQ(im_height, anchor_generator_conf.image_height());
      CHECK_EQ(im_width, anchor_generator_conf.image_width());
    }
    num_anchors += BBoxUtil<MutBBox>::GenerateAnchors(anchor_generator_conf,
                                                      anchors_blob->mut_dptr<T>(num_anchors));
  }
  CHECK_EQ(num_anchors, anchors_blob->shape().At(0));

  IndexSequence inside_inds(num_anchors, inside_inds_blob->mut_dptr<int32_t>(), true);
  inside_inds.Filter([&](int32_t index) {
    const BBox* bbox = BBox::Cast(anchors_blob->dptr<T>(index));
    return bbox->left() < -straddle_thresh || bbox->top() < -straddle_thresh
           || bbox->right() >= im_width + straddle_thresh
           || bbox->bottom() >= im_height + straddle_thresh;
  });
  inside_inds_blob->set_dim0_valid_num(0, inside_inds.size());
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(size_t, im_index, 0, BnInOp2Blob("gt_boxes")->shape().At(0)) {
    auto gt_boxes = GetImageGtBoxes(im_index, BnInOp2Blob);
    auto anchor_boxes = GetImageAnchorBoxes(ctx, im_index, BnInOp2Blob);
    CalcMaxOverlapAndSetPositiveLabels(gt_boxes, anchor_boxes);
    size_t fg_cnt = 0;
    size_t bg_cnt = 0;
    if (op_conf().anchor_target_conf().random_subsample()) {
      fg_cnt = SubsampleForeground(anchor_boxes);
      bg_cnt = SubsampleBackground(fg_cnt, anchor_boxes);
    } else {
      fg_cnt = ChoiceForeground(anchor_boxes);
      bg_cnt = ChoiceBackground(fg_cnt, anchor_boxes);
    }
    OutputForEachImage(im_index, fg_cnt + bg_cnt, gt_boxes, anchor_boxes, BnInOp2Blob);
  }
}

template<typename T>
typename AnchorTargetKernel<T>::MaxOverlapOfLabeledBoxesWithGt
AnchorTargetKernel<T>::GetImageAnchorBoxes(
    const KernelCtx& ctx, size_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* anchor_boxes_inds_blob = BnInOp2Blob("anchor_boxes_inds");
  anchor_boxes_inds_blob->CopyFrom(ctx.device_ctx, BnInOp2Blob("anchors_inside_inds"));
  MaxOverlapOfLabeledBoxesWithGt anchor_boxes(
      MaxOverlapOfBoxesWithGt(
          AnchorBoxes(IndexSequence(anchor_boxes_inds_blob->static_shape().elem_cnt(),
                                    anchor_boxes_inds_blob->dim0_valid_num(0),
                                    anchor_boxes_inds_blob->mut_dptr<int32_t>(), false),
                      BnInOp2Blob("anchors")->dptr<T>()),
          BnInOp2Blob("max_overlaps")->mut_dptr<float>(),
          BnInOp2Blob("max_overlaps_with_gt_index")->mut_dptr<int32_t>(), true),
      BnInOp2Blob("anchor_boxes_labels")->mut_dptr<int32_t>());
  anchor_boxes.FillLabels(-1);
  return anchor_boxes;
}

template<typename T>
typename AnchorTargetKernel<T>::GtBoxes AnchorTargetKernel<T>::GetImageGtBoxes(
    size_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  Blob* gt_boxes_inds_blob = BnInOp2Blob("gt_boxes_inds");
  gt_boxes_inds_blob->set_dim0_valid_num(0, gt_boxes_blob->dim1_valid_num(im_index));
  GtBoxes gt_boxes(IndexSequence(gt_boxes_inds_blob->static_shape().elem_cnt(),
                                 gt_boxes_inds_blob->dim0_valid_num(0),
                                 gt_boxes_inds_blob->mut_dptr<int32_t>(), true),
                   gt_boxes_blob->dptr<T>(im_index));
  gt_boxes.Filter([&](int32_t index) { return gt_boxes.bbox(index)->Area() <= 0; });
  return gt_boxes;
}

template<typename T>
void AnchorTargetKernel<T>::CalcMaxOverlapAndSetPositiveLabels(
    const GtBoxes& gt_boxes, MaxOverlapOfLabeledBoxesWithGt& anchor_boxes) const {
  std::vector<float> gt_max_overlap(gt_boxes.size(), 0);
  std::vector<int32_t> gt_max_overlap_with_indices;
  size_t num_recorded = 0;
  int32_t last_gt_index = -1;
  auto TryRecordGtMaxOverlap = [&](int32_t gt_index, int32_t index, float overlap) {
    if (gt_index != last_gt_index) {
      num_recorded = gt_max_overlap_with_indices.size();
      last_gt_index = gt_index;
    }
    if (overlap >= gt_max_overlap[gt_index]) {
      if (overlap > gt_max_overlap[gt_index]) { gt_max_overlap_with_indices.resize(num_recorded); }
      gt_max_overlap[gt_index] = overlap;
      gt_max_overlap_with_indices.emplace_back(index);
    }
  };
  // Set the anchor box whose max overlap with gt boxes greater than
  // threshold to positive label
  float positive_overlap_threshold = op_conf().anchor_target_conf().positive_overlap_threshold();
  ForEachOverlapBetweenBoxesAndGtBoxes(
      anchor_boxes, gt_boxes, [&](int32_t index, int32_t gt_index, float overlap) {
        anchor_boxes.TryUpdateMaxOverlap(index, gt_index, overlap, [&]() {
          if (overlap >= positive_overlap_threshold) { anchor_boxes.set_label(index, 1); }
        });
        TryRecordGtMaxOverlap(gt_index, index, overlap);
      });
  // Set every anchor box which is nearest to each gt box to positive label
  for (int32_t index : gt_max_overlap_with_indices) { anchor_boxes.set_label(index, 1); }
}

template<typename T>
size_t AnchorTargetKernel<T>::SubsampleForeground(MaxOverlapOfLabeledBoxesWithGt& boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t fg_cnt = conf.total_subsample_num() * conf.foreground_fraction();
  boxes.Sort([&](int32_t lhs_index, int32_t rhs_index) {
    return boxes.label(lhs_index) > boxes.label(rhs_index);
  });
  size_t fg_end = boxes.Find([&](int32_t index) { return boxes.label(index) != 1; });
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
                                                  MaxOverlapOfLabeledBoxesWithGt& boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t bg_cnt = conf.total_subsample_num() - fg_cnt;
  boxes.Sort([&](int32_t lhs_index, int32_t rhs_index) {
    return boxes.max_overlap(lhs_index) < boxes.max_overlap(rhs_index);
  });
  size_t bg_end = boxes.Find(
      [&](int32_t index) { return boxes.max_overlap(index) >= conf.negative_overlap_threshold(); });
  if (bg_end > bg_cnt) {
    boxes.Shuffle(0, bg_end);
  } else {
    bg_cnt = bg_end;
  }
  FOR_RANGE(size_t, i, 0, bg_cnt) { boxes.SetLabel(i, 0); }
  return bg_cnt;
}

template<typename T>
size_t AnchorTargetKernel<T>::ChoiceForeground(MaxOverlapOfLabeledBoxesWithGt& boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t fg_cnt = conf.total_subsample_num() * conf.foreground_fraction();
  size_t fg_num = 0;
  boxes.ForEach([&](int32_t index) {
    int32_t label = boxes.label(index);
    if (label == 1) {
      if (fg_num >= fg_cnt) { boxes.set_label(index, -1); }
      ++fg_num;
    }
    return true;
  });
  return fg_num < fg_cnt ? fg_num : fg_cnt;
}

template<typename T>
size_t AnchorTargetKernel<T>::ChoiceBackground(size_t fg_cnt,
                                               MaxOverlapOfLabeledBoxesWithGt& boxes) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t bg_cnt = conf.total_subsample_num() - fg_cnt;
  size_t bg_num = 0;
  boxes.ForEach([&](int32_t index) {
    if (boxes.max_overlap(index) < conf.negative_overlap_threshold()) {
      if (bg_num < bg_cnt) { boxes.set_label(index, 0); }
      ++bg_num;
    }
    return true;
  });
  return bg_num < bg_cnt ? bg_num : bg_cnt;
}

template<typename T>
void AnchorTargetKernel<T>::OutputForEachImage(
    size_t im_index, size_t total_sample_cnt, const GtBoxes& gt_boxes,
    const MaxOverlapOfLabeledBoxesWithGt& boxes,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  CHECK_GT(total_sample_cnt, 0);
  const float reduction_coefficient = 1.f / total_sample_cnt;
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const BBoxRegressionWeights& bbox_reg_ws = conf.bbox_reg_weights();
  size_t layer_offset = 0;
  FOR_RANGE(size_t, layer, 0, conf.anchor_generator_conf_size()) {
    Blob* rpn_labels_blob = BnInOp2Blob("rpn_labels_" + std::to_string(layer));
    size_t num_per_layer = rpn_labels_blob->shape().Count(1, 4);
    int32_t* rpn_labels_ptr = rpn_labels_blob->mut_dptr<int32_t>(im_index);
    auto* bbox_targets = BBoxDelta<T>::Cast(
        BnInOp2Blob("rpn_bbox_targets_" + std::to_string(layer))->mut_dptr<T>(im_index));
    auto* inside_weights = BBoxWeights<T>::Cast(
        BnInOp2Blob("rpn_bbox_inside_weights_" + std::to_string(layer))->mut_dptr<T>(im_index));
    auto* outside_weights = BBoxWeights<T>::Cast(
        BnInOp2Blob("rpn_bbox_outside_weights_" + std::to_string(layer))->mut_dptr<T>(im_index));
    // Copy label to each layer output
    std::memcpy(rpn_labels_ptr, boxes.label() + layer_offset, num_per_layer * sizeof(int32_t));
    FOR_RANGE(size_t, i, 0, num_per_layer) {
      int32_t bbox_idx = layer_offset + i;
      int32_t label = boxes.label(bbox_idx);
      // Calc bbox target and set inside weights for each layer output
      if (label == 1) {
        const auto* box = boxes.bbox(bbox_idx);
        const auto* gt_box = gt_boxes.GetBBox(boxes.max_overlap_with_index(bbox_idx));
        bbox_targets[i].TransformInverse(box, gt_box, bbox_reg_ws);
        inside_weights[i].set_weight_x(1.f);
        inside_weights[i].set_weight_y(1.f);
        inside_weights[i].set_weight_w(1.f);
        inside_weights[i].set_weight_h(1.f);
      } else {
        bbox_targets[i].set_dx(0.f);
        bbox_targets[i].set_dy(0.f);
        bbox_targets[i].set_dw(0.f);
        bbox_targets[i].set_dh(0.f);
        inside_weights[i].set_weight_x(0.f);
        inside_weights[i].set_weight_y(0.f);
        inside_weights[i].set_weight_w(0.f);
        inside_weights[i].set_weight_h(0.f);
      }
      // Set outside weights for each layer output
      if (label == 1 || label == 0) {
        outside_weights[i].set_weight_x(reduction_coefficient);
        outside_weights[i].set_weight_y(reduction_coefficient);
        outside_weights[i].set_weight_w(reduction_coefficient);
        outside_weights[i].set_weight_h(reduction_coefficient);
      } else {
        outside_weights[i].set_weight_x(0.f);
        outside_weights[i].set_weight_y(0.f);
        outside_weights[i].set_weight_w(0.f);
        outside_weights[i].set_weight_h(0.f);
      }
    }
    layer_offset += num_per_layer;
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
