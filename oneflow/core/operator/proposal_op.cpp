#include "oneflow/core/operator/proposal_op.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void ProposalOp::InitFromOpConf() {
  CHECK(op_conf().has_proposal_conf());
  EnrollInputBn("cls_prob", false);
  EnrollInputBn("bbox_pred", false);
  EnrollOutputBn("rois", false);
  EnrollOutputBn("scores", false);
  EnrollConstBufBn("anchors");
  EnrollDataTmpBn("proposals");
}

const PbMessage& ProposalOp::GetCustomizedConf() const { return op_conf().proposal_conf(); }

void ProposalOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const auto& anchor_scales = GetPbRfFromCustomizedConf<int32_t>("anchor_scales");
  const auto& aspect_ratios = GetPbRfFromCustomizedConf<float>("aspect_ratios");
  int32_t num_of_anchors = anchor_scales.size() * aspect_ratios.size();
  const BlobDesc* cls_prob_blob_desc = GetBlobDesc4BnInOp("cls_prob");
  const BlobDesc* bbox_pred_blob_desc = GetBlobDesc4BnInOp("bbox_pred");
  // bactch
  CHECK_EQ(cls_prob_blob_desc->shape().At(0), bbox_pred_blob_desc->shape().At(0));
  // H
  CHECK_EQ(cls_prob_blob_desc->shape().At(1), bbox_pred_blob_desc->shape().At(1));
  // W
  CHECK_EQ(cls_prob_blob_desc->shape().At(2), bbox_pred_blob_desc->shape().At(2));
  // score 2 * 9
  CHECK_EQ(cls_prob_blob_desc->shape().At(3), 2 * num_of_anchors);
  // proposal 4 * 9
  CHECK_EQ(bbox_pred_blob_desc->shape().At(3), 4 * num_of_anchors);

  // anchors
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->mut_shape() = Shape({num_of_anchors});
  anchors_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());

  // rois
  int32_t num_of_rois = GetValFromCustomizedConf<int32_t>("post_nms_top_n");
  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  rois_blob_desc->mut_shape() = Shape({bbox_pred_blob_desc->shape().At(0), num_of_rois, 5});
  rois_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());

  // scores
  BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("scores");
  scores_blob_desc->mut_shape() = Shape({cls_prob_blob_desc->shape().At(0),
                                         cls_prob_blob_desc->shape().Count(1, 3) * num_of_rois, 1});
  rois_blob_desc->set_data_type(cls_prob_blob_desc->data_type());

  // proposals
  *GetBlobDesc4BnInOp("proposals") = *bbox_pred_blob_desc;
}

}  // namespace oneflow