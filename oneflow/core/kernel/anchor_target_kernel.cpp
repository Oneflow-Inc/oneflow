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
    FOR_RANGE(size_t, i, 0, neg_num) { label_ptr[neg_inds.at(i)] = -1; }
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
  const int32_t batch_height = images_blob->shape().At(1);
  const int32_t batch_width = images_blob->shape().At(2);
  FOR_RANGE(size_t, i, 0, conf.anchor_generator_conf_size()) {
    const AnchorGeneratorConf& anchor_generator_conf = conf.anchor_generator_conf(i);
    const float fm_stride = static_cast<float>(anchor_generator_conf.feature_map_stride());
    Blob* anchors_info_i_blob = BnInOp2Blob("anchors_info_" + std::to_string(i));
    const int32_t fm_height = anchors_info_i_blob->dptr<int32_t>()[0];
    CHECK_EQ(std::ceil(static_cast<float>(batch_height) / fm_stride), fm_height);
    const int32_t fm_width = anchors_info_i_blob->dptr<int32_t>()[1];
    CHECK_EQ(std::ceil(static_cast<float>(batch_width) / fm_stride), fm_width);
    const int32_t num_anchors_per_cell = anchors_info_i_blob->dptr<int32_t>()[2];
    CHECK_EQ(
        anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size(),
        num_anchors_per_cell);
    Shape class_shape({fm_height, fm_width, num_anchors_per_cell});
    Shape regression_shape({fm_height, fm_width, num_anchors_per_cell * 4});
    BnInOp2Blob("regression_targets_" + std::to_string(i))->set_instance_shape(regression_shape);
    BnInOp2Blob("regression_weights_" + std::to_string(i))->set_instance_shape(regression_shape);
    BnInOp2Blob("class_labels_" + std::to_string(i))->set_instance_shape(class_shape);
    BnInOp2Blob("class_weights_" + std::to_string(i))->set_instance_shape(class_shape);
  }
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const Blob* images_blob = BnInOp2Blob("images");
  const int32_t image_height = images_blob->shape().At(1);
  const int32_t image_width = images_blob->shape().At(2);
  FOR_RANGE(int32_t, i, 0, conf.anchor_generator_conf_size()) {
    const auto& anchor_generator_conf = conf.anchor_generator_conf(i);
    const int64_t num_anchors_per_cell =
        anchor_generator_conf.anchor_scales_size() * anchor_generator_conf.aspect_ratios_size();
    const float fm_stride = anchor_generator_conf.feature_map_stride();
    const int64_t height = std::ceil(image_height / fm_stride);
    const int64_t width = std::ceil(image_width / fm_stride);
    BnInOp2Blob("anchors_" + std::to_string(i))
        ->set_dim0_valid_num(0, height * width * num_anchors_per_cell);
  }
}

template<typename T>
void AnchorTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ClearOutputBlobs(ctx.device_ctx, BnInOp2Blob);
  ConcatAnchors(ctx.device_ctx, BnInOp2Blob);
  FOR_RANGE(int64_t, im_index, 0, BnInOp2Blob("images")->shape().At(0)) {
    CalcMaxOverlapAndSetPositiveLabels(ctx.device_ctx, im_index, BnInOp2Blob);
    ExcludeOutsideAnchorBoxes(ctx.device_ctx, im_index, BnInOp2Blob);
    Subsample(ctx.device_ctx, im_index, BnInOp2Blob);
    OutputForEachImage(ctx.device_ctx, im_index, BnInOp2Blob);
  }
}

template<typename T>
void AnchorTargetKernel<T>::ClearOutputBlobs(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  FOR_RANGE(size_t, i, 0, op_conf().anchor_target_conf().anchor_generator_conf_size()) {
    Blob* regression_targets_i_blob = BnInOp2Blob("regression_targets_" + std::to_string(i));
    Blob* regression_weights_i_blob = BnInOp2Blob("regression_weights_" + std::to_string(i));
    Blob* class_labels_i_blob = BnInOp2Blob("class_labels_" + std::to_string(i));
    Blob* class_weights_i_blob = BnInOp2Blob("class_weights_" + std::to_string(i));

    Memset<DeviceType::kCPU>(ctx, regression_targets_i_blob->mut_dptr<T>(), 0,
                             regression_targets_i_blob->ByteSizeOfDataContentField());
    Memset<DeviceType::kCPU>(ctx, regression_weights_i_blob->mut_dptr<T>(), 0,
                             regression_weights_i_blob->ByteSizeOfDataContentField());
    Memset<DeviceType::kCPU>(ctx, class_labels_i_blob->mut_dptr<int32_t>(), 0,
                             class_labels_i_blob->ByteSizeOfDataContentField());
    Memset<DeviceType::kCPU>(ctx, class_weights_i_blob->mut_dptr<T>(), 0,
                             class_weights_i_blob->ByteSizeOfDataContentField());
  }
}

template<typename T>
void AnchorTargetKernel<T>::ConcatAnchors(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* anchor_boxes_blob = BnInOp2Blob("anchor_boxes");
  size_t num_anchors = 0;
  FOR_RANGE(size_t, i, 0, op_conf().anchor_target_conf().anchors_size()) {
    Blob* anchors_i_blob = BnInOp2Blob("anchors_" + std::to_string(i));
    const size_t num_anchors_per_layer = anchors_i_blob->dim0_valid_num(0);
    Memcpy<DeviceType::kCPU>(ctx, anchor_boxes_blob->mut_dptr<T>() + num_anchors * 4 * sizeof(T),
                             anchors_i_blob->dptr<T>(), num_anchors_per_layer * 4 * sizeof(T));
    num_anchors += num_anchors_per_layer;
  }
  CHECK_LE(num_anchors, anchor_boxes_blob->static_shape().At(0));
  anchor_boxes_blob->set_dim0_valid_num(0, num_anchors);
}

template<typename T>
void AnchorTargetKernel<T>::ExcludeOutsideAnchorBoxes(
    DeviceCtx* ctx, int64_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* image_size_blob = BnInOp2Blob("image_size");
  const Blob* anchor_boxes_blob = BnInOp2Blob("anchor_boxes");
  Blob* anchor_labels_blob = BnInOp2Blob("anchor_labels");

  const size_t num_anchors = anchor_boxes_blob->shape().At(0);
  const float im_height = image_size_blob->dptr<int32_t>(im_index)[0];
  const float im_width = image_size_blob->dptr<int32_t>(im_index)[1];
  const float straddle_thresh = op_conf().anchor_target_conf().straddle_thresh();

  const auto* anchor_bbox_ptr = BBox::Cast(anchor_boxes_blob->dptr<T>());
  int32_t* acnhor_labels_ptr = anchor_labels_blob->mut_dptr<int32_t>(im_index);
  FOR_RANGE(size_t, i, 0, num_anchors) {
    const auto& bbox = anchor_bbox_ptr[i];
    if (bbox.left() < -straddle_thresh || bbox.top() < -straddle_thresh
        || bbox.right() >= im_width + straddle_thresh
        || bbox.bottom() >= im_height + straddle_thresh) {
      acnhor_labels_ptr[i] = -2;
    }
  }
}

template<typename T>
void AnchorTargetKernel<T>::CalcMaxOverlapAndSetPositiveLabels(
    DeviceCtx* ctx, int64_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  const Blob* anchor_boxes_blob = BnInOp2Blob("anchor_boxes");
  Blob* anchor_labels_blob = BnInOp2Blob("anchor_labels");
  Blob* anchor_max_overlaps_blob = BnInOp2Blob("anchor_max_overlaps");
  Blob* anchor_best_match_gt_blob = BnInOp2Blob("anchor_best_match_gt");

  size_t num_gt_boxes = gt_boxes_blob->dim1_valid_num(im_index);
  size_t num_anchors = anchor_boxes_blob->dim0_valid_num(0);
  int32_t* anchor_labels_ptr = anchor_labels_blob->mut_dptr<int32_t>(im_index);
  float* anchor_max_overlaps_ptr = anchor_max_overlaps_blob->mut_dptr<float>(im_index);
  int32_t* anchor_best_match_gt_ptr = anchor_best_match_gt_blob->mut_dptr<int32_t>(im_index);

  Memset<DeviceType::kCPU>(ctx, anchor_labels_ptr, 0,
                           anchor_labels_blob->shape().Count(1) * sizeof(int32_t));
  Memset<DeviceType::kCPU>(ctx, anchor_max_overlaps_ptr, 0,
                           anchor_max_overlaps_blob->shape().Count(1) * sizeof(float));
  std::fill(anchor_best_match_gt_ptr,
            anchor_best_match_gt_ptr + anchor_max_overlaps_blob->shape().Count(1), -1);

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
  auto* gt_bbox_ptr = BBox::Cast(gt_boxes_blob->dptr<T>(im_index));
  auto* anchor_bbox_ptr = BBox::Cast(anchor_boxes_blob->dptr<T>());
  FOR_RANGE(size_t, i, 0, num_gt_boxes) {
    FOR_RANGE(size_t, j, 0, num_anchors) {
      float overlap = anchor_bbox_ptr[j].InterOverUnion(gt_bbox_ptr + i);
      if (overlap > anchor_max_overlaps_ptr[j]) {
        anchor_max_overlaps_ptr[j] = overlap;
        anchor_best_match_gt_ptr[j] = i;
        if (overlap >= positive_overlap_threshold) { anchor_labels_ptr[j] = 1; }
      }
      TryFindNearestAnchorsToGtBox(i, j, overlap);
    }
  }
  // Set every anchor box which is nearest to each gt box to positive label
  for (int32_t idx : gt_nearest_anchor_inds) { anchor_labels_ptr[idx] = 1; }
}

template<typename T>
size_t AnchorTargetKernel<T>::Subsample(
    DeviceCtx* ctx, int64_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* anchor_boxes_blob = BnInOp2Blob("anchor_boxes");
  const Blob* anchor_max_overlaps = BnInOp2Blob("anchor_max_overlaps");
  const float* anchor_max_overlaps_ptr = anchor_max_overlaps->dptr<float>(im_index);
  Blob* anchor_labels_blob = BnInOp2Blob("anchor_labels");
  int32_t* anchor_labels_ptr = anchor_labels_blob->mut_dptr<int32_t>(im_index);

  std::vector<int32_t> pos_inds_vec;
  std::vector<int32_t> neg_inds_vec;
  FOR_RANGE(size_t, i, 0, anchor_boxes_blob->dim0_valid_num(0)) {
    if (anchor_labels_ptr[i] == 1) {
      pos_inds_vec.emplace_back(i);
    } else if (anchor_labels_ptr[i] == 0) {
      neg_inds_vec.emplace_back(i);
    }
  }

  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  size_t pos_num = SubsamplePositive(conf.total_subsample_num() * conf.foreground_fraction(),
                                     pos_inds_vec, anchor_labels_ptr, conf.random_subsample());
  size_t neg_num = SubsampleNegative(conf.total_subsample_num() - pos_num, neg_inds_vec,
                                     conf.negative_overlap_threshold(), anchor_max_overlaps_ptr,
                                     anchor_labels_ptr, conf.random_subsample());
  return pos_num + neg_num;
}

template<typename T>
void AnchorTargetKernel<T>::OutputForEachImage(
    DeviceCtx* ctx, int64_t im_index,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const AnchorTargetOpConf& conf = op_conf().anchor_target_conf();
  const T* gt_boxes_ptr = BnInOp2Blob("gt_boxes")->dptr<T>(im_index);
  const T* anchor_boxes_ptr = BnInOp2Blob("anchor_boxes")->dptr<T>();
  const int32_t* anchor_labels_ptr = BnInOp2Blob("anchor_labels")->dptr<int32_t>(im_index);
  const int32_t* anchor_best_match_gt_ptr =
      BnInOp2Blob("anchor_best_match_gt")->dptr<int32_t>(im_index);

  size_t bbox_cnt = 0;
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

    const size_t num_bbox = BnInOp2Blob("anchors_" + std::to_string(i))->dim0_valid_num(0);
    FOR_RANGE(size_t, j, 0, num_bbox) {
      int32_t anchor_idx = bbox_cnt + j;
      int32_t anchor_label = anchor_labels_ptr[anchor_idx];
      if (anchor_label == 1) {
        int32_t gt_idx = anchor_best_match_gt_ptr[anchor_idx];
        CHECK_GE(gt_idx, 0);
        auto* gt_bbox_ptr = BBox::Cast(gt_boxes_ptr) + gt_idx;
        auto* anchor_bbox_ptr = BBox::Cast(anchor_boxes_ptr) + anchor_idx;
        cur_layer_reg_targets[j].TransformInverse(anchor_bbox_ptr, gt_bbox_ptr,
                                                  conf.bbox_reg_weights());
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
            (anchor_label == -1) ? OneVal<T>::value : ZeroVal<T>::value;
      }
    }

    bbox_cnt += num_bbox;
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAnchorTargetConf, AnchorTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
