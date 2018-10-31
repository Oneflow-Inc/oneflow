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
  FOR_RANGE(int64_t, im_i, 0, BnInOp2Blob("class_prob")->shape().At(0)) {
    RegionProposal(im_i, BnInOp2Blob);
    ApplyNms(im_i, BnInOp2Blob);
  }
  WriteRoisToOutput(BnInOp2Blob);
}

template<typename T>
void ProposalKernel<T>::RegionProposal(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalOpConf& conf = op_conf().proposal_conf();
  Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* score_slice_blob = BnInOp2Blob("score_slice");

  size_t num_proposals = proposals_blob->shape().At(1);
  ScoreSlice score_slice(IndexSequence(score_slice_blob->shape().elem_cnt(),
                                       score_slice_blob->mut_dptr<int32_t>(im_index), true),
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
  auto* prop_bbox = MutBBox::Cast(proposals_blob->mut_dptr<T>(im_index));
  FOR_RANGE(size_t, i, 0, score_slice.size()) {
    int32_t index = score_slice.GetIndex(i);
    prop_bbox[i].Transform(anchor_bbox + index, bbox_delta + index, conf.bbox_reg_weights());
    prop_bbox[i].Clip(conf.anchor_generator_conf().image_height(),
                      conf.anchor_generator_conf().image_width());
  }
}

template<typename T>
void ProposalKernel<T>::ApplyNms(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalOpConf& conf = op_conf().proposal_conf();
  const Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* pre_nms_inds_blob = BnInOp2Blob("pre_nms_slice");
  Blob* post_nms_inds_blob = BnInOp2Blob("post_nms_slice");
  BoxesSlice pre_nms_slice(IndexSequence(pre_nms_inds_blob->shape().Count(1),
                                         pre_nms_inds_blob->mut_dptr<int32_t>(im_index), true),
                           proposals_blob->dptr<T>(im_index));
  BoxesSlice post_nms_slice(IndexSequence(post_nms_inds_blob->shape().Count(1),
                                          post_nms_inds_blob->mut_dptr<int32_t>(im_index), false),
                            proposals_blob->dptr<T>(im_index));
  BBoxUtil<BBox>::Nms(conf.nms_threshold(), pre_nms_slice, post_nms_slice);
  post_nms_inds_blob->set_dim1_valid_num(im_index, post_nms_slice.size());
}

template<typename T>
void ProposalKernel<T>::WriteRoisToOutput(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* score_blob = BnInOp2Blob("class_prob");
  const Blob* proposals_blob = BnInOp2Blob("proposals");
  const Blob* score_slice_blob = BnInOp2Blob("score_slice");
  const Blob* post_nms_inds_blob = BnInOp2Blob("post_nms_slice");
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* rois_prob_blob = BnInOp2Blob("roi_probs");
  const int32_t num_images = score_blob->shape().At(0);
  size_t num_output = 0;
  FOR_RANGE(int32_t, i, 0, num_images) {
    size_t post_nms_inds_size = post_nms_inds_blob->dim1_valid_num(i);
    const int32_t* post_nms_inds_ptr = post_nms_inds_blob->dptr<int32_t>(i);
    const auto* prop_bbox = BBox::Cast(proposals_blob->dptr<T>(i));
    auto* roi_bbox = RoiBBox::Cast(rois_blob->mut_dptr<T>(num_output));
    const T* score_ptr = score_blob->dptr<T>(i);
    const int32_t* score_inds_ptr = score_slice_blob->dptr<int32_t>(i);
    T* roi_probs_ptr = rois_prob_blob->mut_dptr<T>(num_output);
    FOR_RANGE(size_t, j, 0, post_nms_inds_size) {
      int32_t index = post_nms_inds_ptr[j];
      roi_bbox[j].set_ltrb(prop_bbox[index].left(), prop_bbox[index].top(),
                           prop_bbox[index].right(), prop_bbox[index].bottom());
      roi_bbox[j].set_index(i);
      roi_probs_ptr[j] = score_ptr[score_inds_ptr[index]];
      rois_blob->set_record_id_in_device_piece(num_output + j, i);
      rois_prob_blob->set_record_id_in_device_piece(num_output + j, i);
    }
    num_output += post_nms_inds_size;
  }
  CHECK_LE(num_output, rois_blob->static_shape().At(0));
  rois_blob->set_dim0_valid_num(0, num_output);
  rois_prob_blob->set_dim0_valid_num(0, num_output);
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
