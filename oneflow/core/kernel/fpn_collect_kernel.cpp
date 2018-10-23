#include "oneflow/core/kernel/fpn_collect_kernel.h"

namespace oneflow {

template<typename T>
void FpnCollectKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const FpnCollectOpConf& conf = this->op_conf().fpn_collect_conf();
  std::vector<const Blob*> rois_fpn_blobs(conf.num_layers());
  std::vector<const Blob*> roi_probs_fpn_blobs(conf.num_layers());
  FOR_RANGE(size_t, i, 0, conf.num_layers()) {
    rois_fpn_blobs[i] = BnInOp2Blob("rpn_rois_fpn_" + std::to_string(i));
    roi_probs_fpn_blobs[i] = BnInOp2Blob("rpn_roi_probs_fpn_" + std::to_string(i));
  }
  Blob* roi_inds_blob = BnInOp2Blob("roi_inds");
  Blob* out_blob = BnInOp2Blob("out");
  size_t num_out_rois = out_blob->static_shape().At(0);

  auto GlobalIndex2LayerAndIndexInLayer = [&](int32_t index) {
    int32_t row_len = roi_inds_blob->shape().At(1);
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

  // TODO: support multi image inference
  IndexSequence roi_inds(roi_inds_blob->shape().elem_cnt(), roi_inds_blob->mut_dptr<int32_t>(),
                         true);
  roi_inds.Filter([&](int32_t index) {
    int32_t layer = -1;
    int32_t index_in_layer = -1;
    std::tie(layer, index_in_layer) = GlobalIndex2LayerAndIndexInLayer(index);
    return index_in_layer >= rois_fpn_blobs[layer]->shape().At(0);
  });
  roi_inds.NthElem(num_out_rois, Compare);
  roi_inds.Truncate(num_out_rois);
  roi_inds.Sort(Compare);

  FOR_RANGE(size_t, i, 0, roi_inds.size()) {
    const auto* roi_bbox = GetRoiBBox(roi_inds.GetIndex(i));
    auto* out_roi_bbox = MutBBox::Cast(out_blob->mut_dptr<T>());
    out_roi_bbox[i].set_corner_coord(roi_bbox->left(), roi_bbox->top(), roi_bbox->right(),
                                     roi_bbox->bottom());
    out_roi_bbox[i].set_index(roi_bbox->index());
    if (out_blob->has_record_idx_in_device_piece_field()) {
      out_blob->set_record_idx_in_device_piece(i, roi_bbox->index());
    }
  }
  out_blob->set_dim0_valid_num(0, roi_inds.size());
}

template<typename T>
void FpnCollectKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(size_t, i, 0, this->op_conf().fpn_collect_conf().num_layers()) {
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
void FpnCollectKernel<T>::ForwardRecordIdxInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FOR_RANGE(size_t, i, 0, this->op_conf().fpn_collect_conf().num_layers()) {
    const Blob* rois_fpn_blobs_i = BnInOp2Blob("rpn_rois_fpn_" + std::to_string(i));
    const Blob* roi_probs_fpn_blobs_i = BnInOp2Blob("rpn_roi_probs_fpn_" + std::to_string(i));
    size_t field_len = rois_fpn_blobs_i->ByteSizeOfRecordIdxInDevicePieceField();
    CHECK_EQ(field_len, roi_probs_fpn_blobs_i->ByteSizeOfRecordIdxInDevicePieceField());
    CHECK_EQ(std::memcmp(rois_fpn_blobs_i->record_idx_in_device_piece_ptr(),
                         roi_probs_fpn_blobs_i->record_idx_in_device_piece_ptr(), field_len),
             0);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnCollectConf, FpnCollectKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
