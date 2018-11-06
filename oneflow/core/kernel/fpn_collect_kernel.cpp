#include "oneflow/core/kernel/fpn_collect_kernel.h"

namespace oneflow {

template<typename T>
void FpnCollectKernel<T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  if (Global<JobDesc>::Get()->IsTrain()) {
    num_groups_ = 1;
    need_group_by_img_ = false;
  } else {
    num_groups_ = Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(*parallel_ctx);
    need_group_by_img_ = true;
  }
}

template<typename T>
void FpnCollectKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const FpnCollectOpConf& conf = this->op_conf().fpn_collect_conf();
  const int32_t num_layers = conf.rpn_rois_fpn_size();
  std::vector<const Blob*> rois_fpn_blobs(num_layers);
  std::vector<const Blob*> roi_probs_fpn_blobs(num_layers);
  int32_t row_len = 0;
  FOR_RANGE(int32_t, i, 0, num_layers) {
    rois_fpn_blobs[i] = BnInOp2Blob("rpn_rois_fpn_" + std::to_string(i));
    roi_probs_fpn_blobs[i] = BnInOp2Blob("rpn_roi_probs_fpn_" + std::to_string(i));
    row_len = std::max<int32_t>(row_len, rois_fpn_blobs[i]->shape().At(0));
  }

  auto GlobalIndex2LayerAndIndexInLayer = [&](int32_t index) {
    int32_t layer = index / row_len;
    int32_t index_in_layer = index % row_len;
    return std::make_pair(layer, index_in_layer);
  };
  auto GetRoiScore = [&](int32_t index) {
    int32_t layer = -1;
    int32_t index_in_layer = -1;
    std::tie(layer, index_in_layer) = GlobalIndex2LayerAndIndexInLayer(index);
    return roi_probs_fpn_blobs[layer]->dptr<T>()[index_in_layer];
  };
  auto GetRoiBBox = [&](int32_t index) {
    int32_t layer = -1;
    int32_t index_in_layer = -1;
    std::tie(layer, index_in_layer) = GlobalIndex2LayerAndIndexInLayer(index);
    return BBox::Cast(rois_fpn_blobs[layer]->dptr<T>()) + index_in_layer;
  };
  auto Compare = [&](int32_t lhs_index, int32_t rhs_index) {
    return GetRoiScore(lhs_index) > GetRoiScore(rhs_index);
  };

  Blob* out_blob = BnInOp2Blob("out");
  auto im_grouped_roi_inds_vec = GroupRoiBoxes(rois_fpn_blobs, row_len);
  auto im_grouped_roi_num_vec = GetRoiGroupSizeVec(im_grouped_roi_inds_vec);
  SingleThreadLoop(im_grouped_roi_inds_vec.size(), [&](int64_t im_i) {
    auto& roi_inds = im_grouped_roi_inds_vec[im_i];
    const size_t rois_num = im_grouped_roi_num_vec[im_i];
    if (rois_num < roi_inds.size()) {
      std::nth_element(roi_inds.begin(), roi_inds.begin() + rois_num, roi_inds.end(), Compare);
      roi_inds.resize(rois_num);
    }
    // TODO: This sort is not required
    std::sort(roi_inds.begin(), roi_inds.end(), Compare);

    size_t beg = 0;
    FOR_RANGE(size_t, i, 0, im_i) { beg += im_grouped_roi_num_vec[i]; }
    size_t rois_cnt = 0;
    for (const int32_t index : roi_inds) {
      const auto* roi_bbox = GetRoiBBox(index);
      auto* out_roi_bbox = MutBBox::Cast(out_blob->mut_dptr<T>());
      const size_t offset = beg + rois_cnt;
      out_roi_bbox[offset].set_ltrb(roi_bbox->left(), roi_bbox->top(), roi_bbox->right(),
                                    roi_bbox->bottom());
      out_roi_bbox[offset].set_index(roi_bbox->index());
      if (out_blob->has_record_id_in_device_piece_field()) {
        out_blob->set_record_id_in_device_piece(offset, roi_bbox->index());
      }
      ++rois_cnt;
    }
    CHECK_EQ(rois_cnt, rois_num);
  });
  size_t total_output_num = 0;
  for (size_t output_num : im_grouped_roi_num_vec) { total_output_num += output_num; }
  CHECK_GE(out_blob->static_shape().At(0), total_output_num);
  out_blob->set_dim0_valid_num(0, total_output_num);
}

template<typename T>
std::vector<std::vector<int32_t>> FpnCollectKernel<T>::GroupRoiBoxes(
    const std::vector<const Blob*>& rois_fpn_blobs, const int32_t row_len) const {
  std::vector<std::vector<int32_t>> im_grouped_roi_inds_vec(num_groups_);
  FOR_RANGE(int32_t, i, 0, rois_fpn_blobs.size()) {
    const auto* rois_fpn_i_bbox = BBox::Cast(rois_fpn_blobs[i]->dptr<T>());
    FOR_RANGE(int32_t, j, 0, rois_fpn_blobs[i]->shape().At(0)) {
      const int32_t index = i * row_len + j;
      const int32_t im_index = rois_fpn_i_bbox[j].index();
      if (need_group_by_img_) {
        im_grouped_roi_inds_vec[im_index].emplace_back(index);
      } else {
        im_grouped_roi_inds_vec.front().emplace_back(index);
      }
    }
  }
  return im_grouped_roi_inds_vec;
}

template<typename T>
std::vector<size_t> FpnCollectKernel<T>::GetRoiGroupSizeVec(
    const std::vector<std::vector<int32_t>>& im_grouped_roi_inds_vec) const {
  std::vector<size_t> im_grouped_roi_size_vec(im_grouped_roi_inds_vec.size());
  const int32_t top_n_per_image = op_conf().fpn_collect_conf().top_n_per_image();
  FOR_RANGE(size_t, i, 0, im_grouped_roi_size_vec.size()) {
    im_grouped_roi_size_vec[i] =
        std::min<size_t>(im_grouped_roi_inds_vec[i].size(), top_n_per_image);
  }
  return im_grouped_roi_size_vec;
}

template<typename T>
void FpnCollectKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(size_t, i, 0, this->op_conf().fpn_collect_conf().rpn_rois_fpn_size()) {
    const Blob* rois_fpn_blobs_i = BnInOp2Blob("rpn_rois_fpn_" + std::to_string(i));
    const Blob* roi_probs_fpn_blobs_i = BnInOp2Blob("rpn_roi_probs_fpn_" + std::to_string(i));
    size_t field_len = rois_fpn_blobs_i->ByteSizeOfDim0ValidNumField();
    CHECK_EQ(field_len, roi_probs_fpn_blobs_i->ByteSizeOfDim0ValidNumField());
    CHECK_EQ(std::memcmp(rois_fpn_blobs_i->dim0_valid_num_ptr(),
                         roi_probs_fpn_blobs_i->dim0_valid_num_ptr(), field_len),
             0);
  }
}

template<typename T>
void FpnCollectKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(size_t, i, 0, this->op_conf().fpn_collect_conf().rpn_rois_fpn_size()) {
    const Blob* rois_fpn_blobs_i = BnInOp2Blob("rpn_rois_fpn_" + std::to_string(i));
    const Blob* roi_probs_fpn_blobs_i = BnInOp2Blob("rpn_roi_probs_fpn_" + std::to_string(i));
    size_t field_len = rois_fpn_blobs_i->ByteSizeOfRecordIdInDevicePieceField();
    CHECK_EQ(field_len, roi_probs_fpn_blobs_i->ByteSizeOfRecordIdInDevicePieceField());
    CHECK_EQ(std::memcmp(rois_fpn_blobs_i->record_id_in_device_piece_ptr(),
                         roi_probs_fpn_blobs_i->record_id_in_device_piece_ptr(), field_len),
             0);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnCollectConf, FpnCollectKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
