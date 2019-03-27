#include "oneflow/core/kernel/proposal_target_kernel.h"

namespace oneflow {

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

template<typename T>
void ProposalTargetKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializeOutputBlob(ctx.device_ctx, BnInOp2Blob);
  GenMatchMatrixBetweenRoiAndGtBoxes(ctx.device_ctx, BnInOp2Blob);
  Subsample(ctx.device_ctx, BnInOp2Blob);
  Output(ctx.device_ctx, BnInOp2Blob);
}

template<typename T>
void ProposalTargetKernel<T>::InitializeOutputBlob(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* sampled_rois_blob = BnInOp2Blob("sampled_rois");
  Blob* sampled_roi_inds_blob = BnInOp2Blob("sampled_roi_inds");
  Blob* class_labels_blob = BnInOp2Blob("class_labels");
  Blob* regression_targets_blob = BnInOp2Blob("regression_targets");
  Blob* regression_weights_blob = BnInOp2Blob("regression_weights");
  Memset<DeviceType::kCPU>(ctx, sampled_rois_blob->mut_dptr(), 0,
                           sampled_rois_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx, sampled_roi_inds_blob->mut_dptr(), 0,
                           sampled_roi_inds_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx, class_labels_blob->mut_dptr(), 0,
                           class_labels_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx, regression_targets_blob->mut_dptr(), 0,
                           regression_targets_blob->ByteSizeOfDataContentField());
  Memset<DeviceType::kCPU>(ctx, regression_weights_blob->mut_dptr(), 0,
                           regression_weights_blob->ByteSizeOfDataContentField());
}

template<typename T>
void ProposalTargetKernel<T>::GenMatchMatrixBetweenRoiAndGtBoxes(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  Blob* max_overlaps_blob = BnInOp2Blob("max_overlaps");
  Blob* best_match_gt_indices_blob = BnInOp2Blob("max_overlaps_with_gt_index");

  const T* rois_ptr = rois_blob->dptr<T>();
  const T* gt_boxes_ptr = gt_boxes_blob->dptr<T>();
  float* max_overlaps_ptr = max_overlaps_blob->mut_dptr<float>();
  int32_t* best_match_gt_indices_ptr = best_match_gt_indices_blob->mut_dptr<int32_t>();
  Memset<DeviceType::kCPU>(ctx, max_overlaps_ptr, 0,
                           max_overlaps_blob->ByteSizeOfDataContentField());
  std::fill(best_match_gt_indices_ptr,
            best_match_gt_indices_ptr + best_match_gt_indices_blob->static_shape().elem_cnt(), -1);

  const size_t num_rois = rois_blob->shape().At(0);
  const size_t max_num_gt_boxes_per_im = gt_boxes_blob->static_shape().At(1);
  MultiThreadLoop(num_rois, [&](int32_t roi_index) {
    auto* roi_bbox = BBox::Cast(rois_ptr) + roi_index;
    const int32_t im_index = roi_bbox->index();
    FOR_RANGE(int32_t, i, 0, gt_boxes_blob->dim1_valid_num(im_index)) {
      int32_t gt_index = im_index * max_num_gt_boxes_per_im + i;
      auto* gt_bbox = GtBBox::Cast(gt_boxes_ptr) + gt_index;
      float overlap = roi_bbox->InterOverUnion(gt_bbox);
      if (overlap > max_overlaps_ptr[roi_index]) {
        max_overlaps_ptr[roi_index] = overlap;
        best_match_gt_indices_ptr[roi_index] = gt_index;
      }
    }
  });
}

template<typename T>
void ProposalTargetKernel<T>::Subsample(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalTargetOpConf& conf = op_conf().proposal_target_conf();
  Blob* sampled_roi_inds_blob = BnInOp2Blob("sampled_roi_inds");
  const size_t num_rois = BnInOp2Blob("rois")->shape().At(0);
  const size_t num_sampled_rois = sampled_roi_inds_blob->static_shape().elem_cnt();
  const float* max_overlaps_ptr = BnInOp2Blob("max_overlaps")->dptr<float>();
  int32_t* best_match_gt_indices_ptr =
      BnInOp2Blob("max_overlaps_with_gt_index")->mut_dptr<int32_t>();
  int32_t* sampled_roi_inds_blob_ptr = sampled_roi_inds_blob->mut_dptr<int32_t>();
  std::vector<int32_t> rois_index_vec(num_rois);
  std::iota(rois_index_vec.begin(), rois_index_vec.end(), 0);
  std::sort(rois_index_vec.begin(), rois_index_vec.end(),
            [=](int32_t lhs_index, int32_t rhs_index) {
              return max_overlaps_ptr[lhs_index] > max_overlaps_ptr[rhs_index];
            });

  // Foregroud sample
  auto fg_low_it = std::find_if(rois_index_vec.begin(), rois_index_vec.end(), [&](int32_t index) {
    return max_overlaps_ptr[index] < conf.foreground_threshold();
  });
  size_t fg_end = std::distance(rois_index_vec.begin(), fg_low_it);
  size_t fg_cnt = num_sampled_rois * conf.foreground_fraction();
  if (fg_cnt < fg_end) {
    if (conf.random_subsample()) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(rois_index_vec.begin(), fg_low_it, gen);
    } else {
      std::sort(rois_index_vec.begin(), fg_low_it, std::less<int32_t>());
    }
  } else {
    fg_cnt = fg_end;
  }
  int32_t* cur_sampled_inds_ptr = sampled_roi_inds_blob_ptr;
  std::copy(rois_index_vec.begin(), rois_index_vec.begin() + fg_cnt, cur_sampled_inds_ptr);
  cur_sampled_inds_ptr += fg_cnt;

  // Backgroud sample
  rois_index_vec.erase(rois_index_vec.begin(), fg_low_it);
  auto bg_high_it = std::find_if(rois_index_vec.begin(), rois_index_vec.end(), [&](int32_t index) {
    return max_overlaps_ptr[index] < conf.background_threshold_high();
  });
  rois_index_vec.erase(rois_index_vec.begin(), bg_high_it);
  auto bg_low_it = std::find_if(rois_index_vec.begin(), rois_index_vec.end(), [&](int32_t index) {
    return max_overlaps_ptr[index] < conf.background_threshold_low();
  });
  rois_index_vec.erase(bg_low_it, rois_index_vec.end());
  size_t bg_cnt = num_sampled_rois - fg_cnt;
  if (bg_cnt < rois_index_vec.size()) {
    if (conf.random_subsample()) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(rois_index_vec.begin(), rois_index_vec.end(), gen);
    } else {
      std::sort(rois_index_vec.begin(), rois_index_vec.end(), std::less<int32_t>());
    }
  } else {
    bg_cnt = rois_index_vec.size();
  }
  rois_index_vec.resize(bg_cnt);
  // Set negative matching gt box index to -1
  for (int32_t index : rois_index_vec) { best_match_gt_indices_ptr[index] = -1; }
  std::copy(rois_index_vec.begin(), rois_index_vec.end(), cur_sampled_inds_ptr);
  size_t total_sampled_cnt = fg_cnt + bg_cnt;

  CHECK_LE(total_sampled_cnt, num_sampled_rois);
  std::sort(sampled_roi_inds_blob_ptr, sampled_roi_inds_blob_ptr + total_sampled_cnt,
            [](int32_t lhs_idx, int32_t rhs_idx) { return lhs_idx < rhs_idx; });
  sampled_roi_inds_blob->set_dim0_valid_num(0, total_sampled_cnt);
}

template<typename T>
void ProposalTargetKernel<T>::Output(
    DeviceCtx* ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  const Blob* gt_labels_blob = BnInOp2Blob("gt_labels");
  const Blob* best_match_gt_indices_blob = BnInOp2Blob("max_overlaps_with_gt_index");
  Blob* sampled_rois_blob = BnInOp2Blob("sampled_rois");
  Blob* sampled_roi_inds_blob = BnInOp2Blob("sampled_roi_inds");
  Blob* class_labels_blob = BnInOp2Blob("class_labels");
  Blob* regression_targets_blob = BnInOp2Blob("regression_targets");
  Blob* regression_weights_blob = BnInOp2Blob("regression_weights");
  const int32_t* sampled_roi_inds_ptr = sampled_roi_inds_blob->dptr<int32_t>();
  auto* rois_bbox_ptr = BBox::Cast(rois_blob->dptr<T>());
  auto* sampled_rois_bbox_ptr = MutBBox::Cast(sampled_rois_blob->mut_dptr<T>());
  auto* gt_bbox_ptr = GtBBox::Cast(gt_boxes_blob->dptr<T>());
  const int32_t* gt_labels_ptr = gt_labels_blob->dptr<int32_t>();
  const int32_t* best_match_gt_indices_ptr = best_match_gt_indices_blob->dptr<int32_t>();
  auto* bbox_regression_ptr = BBoxDelta<T>::Cast(regression_targets_blob->mut_dptr<T>());
  auto* regression_weights_ptr = BBoxWeights<T>::Cast(regression_weights_blob->mut_dptr<T>());
  int32_t* roi_labels_ptr = class_labels_blob->mut_dptr<int32_t>();

  const size_t num_out_rois = sampled_roi_inds_blob->shape().elem_cnt();
  MultiThreadLoop(num_out_rois, [&](int32_t i) {
    const int32_t index = sampled_roi_inds_ptr[i];
    auto* rois_bbox = rois_bbox_ptr + index;
    const int32_t im_index = rois_bbox->index();
    sampled_rois_bbox_ptr[i].set_ltrb(rois_bbox->left(), rois_bbox->top(), rois_bbox->right(),
                                      rois_bbox->bottom());
    sampled_rois_bbox_ptr[i].set_index(im_index);

    const int32_t gt_index = best_match_gt_indices_ptr[index];
    if (gt_index >= 0) {
      roi_labels_ptr[i] = gt_labels_ptr[gt_index];
      bbox_regression_ptr[i].TransformInverse(rois_bbox, gt_bbox_ptr + gt_index,
                                              op_conf().proposal_target_conf().bbox_reg_weights());
      regression_weights_ptr[i].set_weight_x(OneVal<T>::value);
      regression_weights_ptr[i].set_weight_y(OneVal<T>::value);
      regression_weights_ptr[i].set_weight_w(OneVal<T>::value);
      regression_weights_ptr[i].set_weight_h(OneVal<T>::value);
    } else {
      roi_labels_ptr[i] = 0;
    }
    if (rois_blob->has_record_id_in_device_piece_field()) {
      sampled_rois_blob->set_record_id_in_device_piece(i, im_index);
      sampled_roi_inds_blob->set_record_id_in_device_piece(i, im_index);
      class_labels_blob->set_record_id_in_device_piece(i, im_index);
      regression_targets_blob->set_record_id_in_device_piece(i, im_index);
      regression_weights_blob->set_record_id_in_device_piece(i, im_index);
    }
  });
  sampled_rois_blob->set_dim0_valid_num(0, num_out_rois);
  sampled_roi_inds_blob->set_dim0_valid_num(0, num_out_rois);
  class_labels_blob->set_dim0_valid_num(0, num_out_rois);
  regression_targets_blob->set_dim0_valid_num(0, num_out_rois);
  regression_weights_blob->set_dim0_valid_num(0, num_out_rois);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalTargetConf, ProposalTargetKernel,
                               FLOATING_DATA_TYPE_SEQ);
}  // namespace oneflow
