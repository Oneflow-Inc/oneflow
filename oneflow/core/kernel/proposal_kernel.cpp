#include "oneflow/core/kernel/proposal_kernel.h"

namespace oneflow {

template<typename T>
void ProposalKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BBoxUtil<MutBBox>::GenerateAnchors(op_conf().proposal_conf().anchor_generator_conf(),
                                     BnInOp2Blob("anchors")->mut_dptr<T>());
}

template<typename T>
void ProposalKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int32_t num_output = 0;
  FOR_RANGE(int64_t, im_i, 0, BnInOp2Blob("class_prob")->shape().At(0)) {
    auto score_slice = RegionProposal(im_i, BnInOp2Blob);
    auto post_nms_slice = ApplyNms(BnInOp2Blob);
    num_output += WriteRoisToOutput(num_output, im_i, score_slice, post_nms_slice, BnInOp2Blob);
  }
  BnInOp2Blob("rois")->set_dim0_valid_num(0, num_output);
  BnInOp2Blob("roi_probs")->set_dim0_valid_num(0, num_output);
}

template<typename T>
typename ProposalKernel<T>::ScoreSlice ProposalKernel<T>::RegionProposal(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalOpConf& conf = op_conf().proposal_conf();
  Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* score_slice_blob = BnInOp2Blob("score_slice");

  size_t num_proposals = proposals_blob->shape().At(0);
  ScoreSlice score_slice(IndexSequence(score_slice_blob->shape().elem_cnt(),
                                       score_slice_blob->mut_dptr<int32_t>(), true),
                         BnInOp2Blob("class_prob")->dptr<T>(im_index));
  score_slice.NthElem(num_proposals, [&](int32_t lhs_index, int32_t rhs_index) {
    return score_slice.score(lhs_index) > score_slice.score(rhs_index);
  });
  score_slice.Truncate(num_proposals);
  score_slice.Sort([&](int32_t lhs_index, int32_t rhs_index) {
    return score_slice.score(lhs_index) > score_slice.score(rhs_index);
  });
  const auto* bbox_delta = BBoxDelta<T>::Cast(BnInOp2Blob("bbox_pred")->dptr<T>(im_index));
  const auto* anchor_bbox = BBox::Cast(BnInOp2Blob("anchors")->dptr<T>());
  auto* prop_bbox = MutBBox::Cast(proposals_blob->mut_dptr<T>());
  FOR_RANGE(size_t, i, 0, score_slice.size()) {
    int32_t index = score_slice.GetIndex(i);
    prop_bbox[i].Transform(anchor_bbox + index, bbox_delta + index, conf.bbox_reg_weights());
    prop_bbox[i].Clip(conf.anchor_generator_conf().image_height(),
                      conf.anchor_generator_conf().image_width());
  }
  return score_slice;
}

template<typename T>
typename ProposalKernel<T>::BoxesSlice ProposalKernel<T>::ApplyNms(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalOpConf& conf = op_conf().proposal_conf();
  const Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* pre_nms_inds_blob = BnInOp2Blob("pre_nms_slice");
  Blob* post_nms_inds_blob = BnInOp2Blob("post_nms_slice");
  BoxesSlice pre_nms_slice(IndexSequence(pre_nms_inds_blob->shape().elem_cnt(),
                                         pre_nms_inds_blob->mut_dptr<int32_t>(), true),
                           proposals_blob->dptr<T>());
  BoxesSlice post_nms_slice(IndexSequence(post_nms_inds_blob->shape().elem_cnt(),
                                          post_nms_inds_blob->mut_dptr<int32_t>(), false),
                            proposals_blob->dptr<T>());
  BBoxUtil<BBox>::Nms(conf.nms_threshold(), pre_nms_slice, post_nms_slice);
  return post_nms_slice;
}

template<typename T>
size_t ProposalKernel<T>::WriteRoisToOutput(
    const size_t num_output, const int32_t im_index, const ScoreSlice& score_slice,
    const BoxesSlice& post_nms_slice,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* rois_prob_blob = BnInOp2Blob("roi_probs");
  FOR_RANGE(size_t, i, 0, post_nms_slice.size()) {
    const auto* prop_bbox = post_nms_slice.GetBBox(i);
    auto* roi_bbox = RoiBBox::Cast(rois_blob->mut_dptr<T>(num_output));
    roi_bbox[i].set_ltrb(prop_bbox->left(), prop_bbox->top(), prop_bbox->right(),
                         prop_bbox->bottom());
    roi_bbox[i].set_index(im_index);
    rois_prob_blob->mut_dptr<T>(num_output)[i] = score_slice.GetScore(post_nms_slice.GetIndex(i));
    rois_blob->set_record_id_in_device_piece(num_output + i, im_index);
    rois_prob_blob->set_record_id_in_device_piece(num_output + i, im_index);
  }
  return post_nms_slice.size();
}

template<typename T>
void ProposalKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void ProposalKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalConf, ProposalKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
