#include "oneflow/core/kernel/anchor_target_kernel.h"

namespace oneflow {

namespace {

size_t SubsamplePositive(size_t pos_cnt, std::vector<int32_t>& pos_inds, int32_t* label_ptr,
                         bool shuffle = true) {
  size_t pos_num = pos_cnt;
  if (shuffle) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(pos_inds.begin(), pos_inds.end(), gen);
  }
  if (pos_cnt < pos_inds.size()) {
    FOR_RANGE(size_t, i, pos_cnt, pos_inds.size()) {
      int32_t index = pos_inds.at(i);
      label_ptr[index] = -1;
    }
  } else {
    pos_num = pos_inds.size();
  }
  return pos_num;
}

size_t SubsampleNegative(size_t neg_cnt, std::vector<int32_t>& neg_inds, float threshold,
                         const float* overlap_ptr, int32_t* label_ptr, bool shuffle = true) {
  size_t neg_num = neg_cnt;
  std::remove_if(neg_inds.begin(), neg_inds.end(),
                 [=](int32_t index) { return overlap_ptr[index] >= threshold; });
  if (shuffle) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(neg_inds.begin(), neg_inds.end(), gen);
  }
  if (neg_num < neg_inds.size()) {
    FOR_RANGE(size_t, i, 0, neg_num) { label_ptr[neg_inds.at(i)] = 0; }
  } else {
    neg_num = neg_inds.size();
  }
  return neg_num;
}

}  // namespace

template<typename T>
void AnchorTargetKernel<T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const Blob* images_blob = BnInOp2Blob("images");
  const int32_t image_height = images_blob->shape().At(1);
  const int32_t image_width = images_blob->shape().At(2);
  FOR_RANGE(size_t, i, 0, conf.anchor_generator_conf_size()) {
    const auto& anchor_generator_conf = conf.anchor_generator_conf(i);
    const int64_t num_anchors_per_cell =
        anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size();
    const float fm_stride = anchor_generator_conf.feature_map_stride();
    const int64_t height = std::ceil(image_height / fm_stride);
    const int64_t width = std::ceil(image_width / fm_stride);
    Shape class_shape({height, width, num_anchors_per_cell});
    Shape regression_shape({height, width, num_anchors_per_cell * 4});
    BnInOp2Blob("regression_targets_" + std::to_string(i))->set_instance_shape(regression_shape);
    BnInOp2Blob("regression_weights_" + std::to_string(i))->set_instance_shape(regression_shape);
    BnInOp2Blob("class_labels_" + std::to_string(i))->set_instance_shape(class_shape);
    BnInOp2Blob("class_weights_" + std::to_string(i))->set_instance_shape(class_shape);
  }
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GenerateScaledGtBoxes(ctx.device_ctx, BnInOp2Blob);
  GenerateAnchorBoxes(ctx.device_ctx, BnInOp2Blob);
  FilterOutsideAnchorBoxes(ctx.device_ctx, BnInOp2Blob);
  FOR_RANGE(size_t, im_index, 0, BnInOp2Blob("images")->shape().At(0)) {
    CalcMaxOverlapAndSetPositiveLabels(ctx.device_ctx, im_index, BnInOp2Blob);
    Subsample(ctx.device_ctx, im_index, BnInOp2Blob);
    OutputForEachImage(ctx.device_ctx, im_index, BnInOp2Blob);
  }
}

template<typename T>
void AnchorTargetKernel<T>::GenerateAnchorBoxes(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const Blob* images_blob = BnInOp2Blob("images");  // shape (N, H, W, C)
  Blob* anchors_blob = BnInOp2Blob("anchors");
  const int64_t im_height = images_blob->shape().At(1);
  const int64_t im_width = images_blob->shape().At(2);
  size_t num_anchors = 0;
  for (const AnchorGeneratorConf& anchor_generator_conf : conf.anchor_generator_conf()) {
    float fm_stride = anchor_generator_conf.feature_map_stride();
    auto scales_vec = PbRf2StdVec(anchor_generator_conf.anchor_scales());
    auto ratios_vec = PbRf2StdVec(anchor_generator_conf.aspect_ratios());
    num_anchors +=
        BBoxUtil<MutBBox>::GenerateAnchors(im_height, im_width, fm_stride, scales_vec, ratios_vec,
                                           anchors_blob->mut_dptr<T>(num_anchors));
  }
  CHECK_LE(num_anchors, anchors_blob->static_shape().At(0));
  anchors_blob->set_dim0_valid_num(0, num_anchors);
}

template<typename T>
void AnchorTargetKernel<T>::FilterOutsideAnchorBoxes(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* images_blob = BnInOp2Blob("images");
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  Blob* anchor_inds_blob = BnInOp2Blob("anchor_inds");
  const size_t num_anchors = anchors_blob->shape().At(0);
  const float im_height = images_blob->shape().At(1);
  const float im_width = images_blob->shape().At(2);
  const float straddle_thresh = op_conf().anchor_target_conf().straddle_thresh();

  size_t valid_anchors_cnt = 0;
  int32_t* anchor_inds_ptr = anchor_inds_blob->mut_dptr<int32_t>();
  auto* anchor_box = BBox::Cast(anchors_blob->dptr<T>());
  FOR_RANGE(size_t, i, 0, num_anchors) {
    if (anchor_box[i].left() >= -straddle_thresh && anchor_box[i].top() >= -straddle_thresh
        && anchor_box[i].right() < im_width + straddle_thresh
        && anchor_box[i].bottom() < im_height + straddle_thresh) {
      anchor_inds_ptr[valid_anchors_cnt] = i;
      valid_anchors_cnt += 1;
    }
  }
  anchor_inds_blob->set_dim0_valid_num(0, valid_anchors_cnt);
}

template<typename T>
void AnchorTargetKernel<T>::GenerateScaledGtBoxes(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  const Blob* im_scale_blob = BnInOp2Blob("im_scale");
  Blob* gt_boxes_scaled_blob = BnInOp2Blob("gt_boxes_scaled");

  FOR_RANGE(size_t, i, 0, gt_boxes_blob->shape().At(0)) {
    size_t valid_gt_boxes_cnt_per_im = 0;
    auto* gt_boxes = BBox::Cast(gt_boxes_blob->dptr<T>(i));
    auto* gt_scaled_boxes = BBox::Cast(gt_boxes_scaled_blob->mut_dptr<T>(i));
    const T scale = im_scale_blob->dptr<T>()[i];
    FOR_RANGE(size_t, j, 0, gt_boxes_blob->dim1_valid_num(i)) {
      if (gt_boxes[j].Area() > 0) {
        gt_scaled_boxes[valid_gt_boxes_cnt_per_im].set_ltrb(
            gt_boxes[j].left() * scale, gt_boxes[j].top() * scale, gt_boxes[j].right() * scale,
            gt_boxes[j].bottom() * scale);
        valid_gt_boxes_cnt_per_im += 1;
      }
    }
    gt_boxes_scaled_blob->set_dim1_valid_num(i, valid_gt_boxes_cnt_per_im);
  }
}

template<typename T>
void AnchorTargetKernel<T>::CalcMaxOverlapAndSetPositiveLabels(
    DeviceCtx* ctx, size_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes_scaled");
  size_t num_gt_boxes = gt_boxes_blob->dim1_valid_num(im_index);
  std::vector<float> gt_max_overlaps(num_gt_boxes, 0);
  std::vector<int32_t> gt_nearest_anchor_inds;
  size_t num_recorded = 0;
  int32_t last_gt_index = -1;
  auto TryFindNearestAnchorsToGtBox = [&](int32_t gt_index, int32_t index, float overlap) {
    if (gt_index != last_gt_index) {
      num_recorded = gt_nearest_anchor_inds.size();
      last_gt_index = gt_index;
    }
    if (overlap >= gt_max_overlaps[gt_index]) {
      if (overlap > gt_max_overlaps[gt_index]) { gt_nearest_anchor_inds.resize(num_recorded); }
      gt_max_overlaps[gt_index] = overlap;
      gt_nearest_anchor_inds.emplace_back(index);
    }
  };
  // Set the anchor box whose max overlap with gt boxes greater than
  // threshold to positive label
  const float positive_overlap_threshold =
      op_conf().anchor_target_conf().positive_overlap_threshold();
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  const Blob* anchor_inds_blob = BnInOp2Blob("anchor_inds");
  Blob* anchor_labels_blob = BnInOp2Blob("anchor_labels");
  Blob* anchor_max_overlaps_blob = BnInOp2Blob("anchor_max_overlaps");
  Blob* anchor_best_match_gt_blob = BnInOp2Blob("anchor_best_match_gt");
  anchor_labels_blob->set_dim0_valid_num(0, anchors_blob->dim0_valid_num(0));
  const int32_t* anchor_inds_ptr = anchor_inds_blob->dptr<int32_t>();
  int32_t* anchor_labels_ptr = anchor_labels_blob->mut_dptr<int32_t>();
  float* anchor_max_overlaps_ptr = anchor_max_overlaps_blob->mut_dptr<float>();
  int32_t* anchor_best_match_gt_ptr = anchor_best_match_gt_blob->mut_dptr<int32_t>();
  Memset<DeviceType::kCPU>(ctx, anchor_max_overlaps_ptr, 0,
                           anchor_max_overlaps_blob->ByteSizeOfDataContentField());
  std::fill(anchor_labels_ptr, anchor_labels_ptr + anchor_labels_blob->shape().elem_cnt(), -1);
  std::fill(anchor_best_match_gt_ptr,
            anchor_best_match_gt_ptr + anchor_max_overlaps_blob->shape().elem_cnt(), -1);

  auto* gt_boxes = BBox::Cast(gt_boxes_blob->dptr<T>());
  auto* anchor_boxes = BBox::Cast(anchors_blob->dptr<T>());
  FOR_RANGE(size_t, i, 0, num_gt_boxes) {
    FOR_RANGE(size_t, j, 0, anchor_inds_blob->dim0_valid_num(0)) {
      int32_t anchor_idx = anchor_inds_ptr[j];
      float overlap = anchor_boxes[anchor_idx].InterOverUnion(gt_boxes + i);
      if (overlap > anchor_max_overlaps_ptr[anchor_idx]) {
        anchor_max_overlaps_ptr[anchor_idx] = overlap;
        anchor_best_match_gt_ptr[anchor_idx] = i;
        if (overlap >= positive_overlap_threshold) { anchor_labels_ptr[anchor_idx] = 1; }
      }
      TryFindNearestAnchorsToGtBox(i, anchor_idx, overlap);
    }
  }
  // Set every anchor box which is nearest to each gt box to positive label
  for (int32_t idx : gt_nearest_anchor_inds) { anchor_labels_ptr[idx] = 1; }
}

template<typename T>
size_t AnchorTargetKernel<T>::Subsample(
    DeviceCtx* ctx, size_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  std::vector<int32_t> pos_inds_vec;
  std::vector<int32_t> neg_inds_vec;
  const Blob* anchor_inds_blob = BnInOp2Blob("anchor_inds");
  const int32_t* anchor_inds_ptr = anchor_inds_blob->dptr<int32_t>();
  int32_t* anchor_labels_ptr = BnInOp2Blob("anchor_labels")->mut_dptr<int32_t>();
  FOR_RANGE(size_t, i, 0, anchor_inds_blob->dim0_valid_num(0)) {
    int32_t anchor_idx = anchor_inds_ptr[i];
    if (anchor_labels_ptr[anchor_idx] == 1) {
      pos_inds_vec.emplace_back(anchor_idx);
    } else {
      neg_inds_vec.emplace_back(anchor_idx);
    }
  }

  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t pos_num = SubsamplePositive(conf.total_subsample_num() * conf.foreground_fraction(),
                                     pos_inds_vec, anchor_labels_ptr, conf.random_subsample());
  size_t neg_num = SubsampleNegative(conf.total_subsample_num() - pos_num, neg_inds_vec,
                                     conf.negative_overlap_threshold(),
                                     BnInOp2Blob("anchor_max_overlaps")->dptr<float>(),
                                     anchor_labels_ptr, conf.random_subsample());
  return pos_num + neg_num;
}

template<typename T>
void AnchorTargetKernel<T>::OutputForEachImage(
    DeviceCtx* ctx, size_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const T* gt_boxes_ptr = BnInOp2Blob("gt_boxes_scaled")->dptr<T>(im_index);
  const T* anchors_ptr = BnInOp2Blob("anchors")->dptr<T>(im_index);
  const int32_t* cur_image_anchor_labels_ptr =
      BnInOp2Blob("anchor_labels")->dptr<int32_t>(im_index);
  const int32_t* anchor_best_match_gt_ptr =
      BnInOp2Blob("anchor_best_match_gt")->dptr<int32_t>(im_index);
  size_t layer_offset = 0;
  FOR_RANGE(size_t, i, 0, conf.anchor_generator_conf_size()) {
    Blob* regression_targets_i_blob = BnInOp2Blob("regression_targets_" + std::to_string(i));
    Blob* regression_weights_i_blob = BnInOp2Blob("regression_weights_" + std::to_string(i));
    Blob* class_labels_i_blob = BnInOp2Blob("class_labels_" + std::to_string(i));
    Blob* class_weights_i_blob = BnInOp2Blob("class_weights_" + std::to_string(i));
    int32_t* cur_layer_class_labels_ptr = class_labels_i_blob->mut_dptr<int32_t>(im_index);
    T* cur_layer_class_weights_ptr = class_weights_i_blob->mut_dptr<T>(im_index);
    auto* cur_layer_reg_targets =
        BBoxDelta<T>::Cast(regression_targets_i_blob->mut_dptr<T>(im_index));
    auto* cur_layer_reg_weights =
        BBoxWeights<T>::Cast(regression_weights_i_blob->mut_dptr<T>(im_index));

    const size_t num_per_layer = class_labels_i_blob->shape().Count(1, 4);
    FOR_RANGE(size_t, j, 0, num_per_layer) {
      int32_t anchor_idx = layer_offset + j;
      int32_t anchor_label = cur_image_anchor_labels_ptr[anchor_idx];
      if (anchor_label == 1) {
        int32_t gt_idx = anchor_best_match_gt_ptr[anchor_idx];
        CHECK_GE(gt_idx, 0);
        auto* gt_box = BBox::Cast(gt_boxes_ptr) + gt_idx;
        auto* anchor_box = BBox::Cast(anchors_ptr) + anchor_idx;
        cur_layer_reg_targets[j].TransformInverse(anchor_box, gt_box, conf.bbox_reg_weights());
        cur_layer_reg_weights[j].set_weight_x(OneVal<T>::value);
        cur_layer_reg_weights[j].set_weight_y(OneVal<T>::value);
        cur_layer_reg_weights[j].set_weight_w(OneVal<T>::value);
        cur_layer_reg_weights[j].set_weight_h(OneVal<T>::value);
        cur_layer_class_labels_ptr[j] = 1;
        cur_layer_class_weights_ptr[j] = OneVal<T>::value;
      } else {
        cur_layer_reg_targets[j].set_dx(ZeroVal<T>::value);
        cur_layer_reg_targets[j].set_dy(ZeroVal<T>::value);
        cur_layer_reg_targets[j].set_dw(ZeroVal<T>::value);
        cur_layer_reg_targets[j].set_dh(ZeroVal<T>::value);
        cur_layer_reg_weights[j].set_weight_x(ZeroVal<T>::value);
        cur_layer_reg_weights[j].set_weight_y(ZeroVal<T>::value);
        cur_layer_reg_weights[j].set_weight_w(ZeroVal<T>::value);
        cur_layer_reg_weights[j].set_weight_h(ZeroVal<T>::value);
        cur_layer_class_labels_ptr[j] = 0;
        cur_layer_class_weights_ptr[j] =
            (anchor_label == -1) ? ZeroVal<T>::value : OneVal<T>::value;
      }
    }
    layer_offset += num_per_layer;
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
