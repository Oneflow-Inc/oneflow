#include "oneflow/core/kernel/proposal_kernel.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

template<typename T>
void ProposalKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  MultiThreadLoop(BnInOp2Blob("class_prob")->shape().At(0), [&](int64_t im_i) {
    RegionProposal(im_i, BnInOp2Blob);
    ApplyNms(im_i, BnInOp2Blob);
  });
  Output(BnInOp2Blob);
}

template<typename T>
void ProposalKernel<T>::RegionProposal(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalOpConf& conf = op_conf().proposal_conf();
  const Blob* class_prob_blob = BnInOp2Blob("class_prob");
  const Blob* bbox_pred_blob = BnInOp2Blob("bbox_pred");
  const Blob* anchors_blob = BnInOp2Blob("anchors");
  const Blob* image_size_blob = BnInOp2Blob("image_size");
  Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* proposal_inds_blob = BnInOp2Blob("proposal_inds");

  const int64_t fm_height = class_prob_blob->shape().At(1);
  const int64_t fm_width = class_prob_blob->shape().At(2);
  const int64_t num_anchors_per_fm = class_prob_blob->shape().At(3);
  const int64_t im_height = image_size_blob->dptr<int32_t>(im_index)[0];
  const int64_t im_width = image_size_blob->dptr<int32_t>(im_index)[1];
  const size_t num_anchors = anchors_blob->dim0_valid_num(0);
  CHECK_EQ(fm_height, bbox_pred_blob->shape().At(1));
  CHECK_EQ(fm_width, bbox_pred_blob->shape().At(2));
  CHECK_EQ(num_anchors_per_fm, bbox_pred_blob->shape().At(3));
  CHECK_EQ(num_anchors, fm_height * fm_width * num_anchors_per_fm);

  const T* score_ptr = class_prob_blob->dptr<T>(im_index);
  std::vector<int32_t> proposal_inds_vec(num_anchors);
  std::iota(proposal_inds_vec.begin(), proposal_inds_vec.end(), 0);
  std::sort(
      proposal_inds_vec.begin(), proposal_inds_vec.end(),
      [=](int32_t lhs_idx, int32_t rhs_idx) { return score_ptr[lhs_idx] > score_ptr[rhs_idx]; });
  size_t num_proposals = std::min<size_t>(proposals_blob->static_shape().At(1), num_anchors);
  proposal_inds_vec.resize(num_proposals);

  auto* reg_ptr = BBoxDelta<T>::Cast(bbox_pred_blob->dptr<T>(im_index));
  auto* anchor_ptr = BBox::Cast(anchors_blob->dptr<T>());
  auto* proposal_ptr = MutBBox::Cast(proposals_blob->mut_dptr<T>(im_index));
  for (size_t i = 0; i < proposal_inds_vec.size(); ++i) {
    int32_t idx = proposal_inds_vec.at(i);
    proposal_ptr[i].Transform(anchor_ptr + idx, reg_ptr + idx, conf.bbox_reg_weights());
    proposal_ptr[i].Clip(im_height, im_width);
  }

  std::copy(proposal_inds_vec.begin(), proposal_inds_vec.end(),
            proposal_inds_blob->mut_dptr<int32_t>(im_index));
  proposal_inds_blob->set_dim1_valid_num(im_index, num_proposals);
  proposals_blob->set_dim1_valid_num(im_index, num_proposals);
}

template<typename T>
void ProposalKernel<T>::ApplyNms(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const ProposalOpConf& conf = op_conf().proposal_conf();
  const Blob* proposals_blob = BnInOp2Blob("proposals");
  Blob* pre_nms_inds_blob = BnInOp2Blob("pre_nms_slice");
  Blob* post_nms_inds_blob = BnInOp2Blob("post_nms_slice");
  BoxesSlice pre_nms_slice(IndexSequence(proposals_blob->dim1_valid_num(im_index),
                                         pre_nms_inds_blob->mut_dptr<int32_t>(im_index), true),
                           proposals_blob->dptr<T>(im_index));
  BoxesSlice post_nms_slice(IndexSequence(post_nms_inds_blob->shape().Count(1),
                                          post_nms_inds_blob->mut_dptr<int32_t>(im_index), false),
                            proposals_blob->dptr<T>(im_index));
  BBoxUtil<BBox>::Nms(conf.nms_threshold(), pre_nms_slice, post_nms_slice);
  post_nms_inds_blob->set_dim1_valid_num(im_index, post_nms_slice.size());
}

template<typename T>
void ProposalKernel<T>::Output(const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* score_blob = BnInOp2Blob("class_prob");
  const Blob* proposals_blob = BnInOp2Blob("proposals");
  const Blob* proposal_inds_blob = BnInOp2Blob("proposal_inds");
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
    const int32_t* proposal_inds_ptr = proposal_inds_blob->dptr<int32_t>(i);
    T* roi_probs_ptr = rois_prob_blob->mut_dptr<T>(num_output);
    FOR_RANGE(size_t, j, 0, post_nms_inds_size) {
      int32_t index = post_nms_inds_ptr[j];
      roi_bbox[j].set_ltrb(prop_bbox[index].left(), prop_bbox[index].top(),
                           prop_bbox[index].right(), prop_bbox[index].bottom());
      roi_bbox[j].set_index(i);
      roi_probs_ptr[j] = score_ptr[proposal_inds_ptr[index]];
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
