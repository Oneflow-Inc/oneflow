#include "oneflow/core/operator/proposal_op.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void ProposalOp::InitFromOpConf() {
  CHECK(op_conf().has_proposal_conf());
  EnrollInputBn("class_prob", false);
  EnrollInputBn("bbox_pred", false);
  EnrollInputBn("image_info", false);
  EnrollOutputBn("rois", false);
  EnrollOutputBn("scores", false);
  EnrollConstBufBn("anchors");
  EnrollDataTmpBn("fg_scores");
  EnrollDataTmpBn("proposals");
}

const PbMessage& ProposalOp::GetCustomizedConf() const { return op_conf().proposal_conf(); }

void ProposalOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const auto& anchor_scales = GetPbRfFromCustomizedConf<int32_t>("anchor_scales");
  const auto& aspect_ratios = GetPbRfFromCustomizedConf<float>("aspect_ratios");
  int32_t num_of_anchors = anchor_scales.size() * aspect_ratios.size();
  int32_t feature_map_stride = GetValFromCustomizedConf<int32_t>("feature_map_stride");
  for (int32_t scale : anchor_scales) {
    CHECK_GT(scale, feature_map_stride);
    CHECK_EQ(scale % feature_map_stride, 0);
  }

  int32_t pre_nms_top_n = GetValFromCustomizedConf<int32_t>("pre_nms_top_n");
  int32_t post_nms_top_n = GetValFromCustomizedConf<int32_t>("post_nms_top_n");
  CHECK_GT(post_nms_top_n, 0);
  CHECK(pre_nms_top_n == -1 || pre_nms_top_n > post_nms_top_n);

  const BlobDesc* cls_prob_blob_desc = GetBlobDesc4BnInOp("class_prob");
  const BlobDesc* bbox_pred_blob_desc = GetBlobDesc4BnInOp("bbox_pred");
  const BlobDesc* im_info_blob_desc = GetBlobDesc4BnInOp("image_info");
  // bactch
  CHECK_EQ(cls_prob_blob_desc->shape().At(0), bbox_pred_blob_desc->shape().At(0));
  CHECK_EQ(cls_prob_blob_desc->shape().At(0), im_info_blob_desc->shape().At(0));
  // H
  CHECK_EQ(cls_prob_blob_desc->shape().At(1), bbox_pred_blob_desc->shape().At(1));
  // W
  CHECK_EQ(cls_prob_blob_desc->shape().At(2), bbox_pred_blob_desc->shape().At(2));
  // score 2 * 9
  CHECK_EQ(cls_prob_blob_desc->shape().At(3), 2 * num_of_anchors);
  // proposal 4 * 9
  CHECK_EQ(bbox_pred_blob_desc->shape().At(3), 4 * num_of_anchors);
  // im_info (n, (origin_height, origin_width, scale))
  CHECK_EQ(im_info_blob_desc->shape().At(1), 3);

  // anchors
  BlobDesc* anchors_blob_desc = GetBlobDesc4BnInOp("anchors");
  anchors_blob_desc->mut_shape() = Shape({num_of_anchors});
  anchors_blob_desc->set_data_type(DataType::kInt32);

  // rois
  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  rois_blob_desc->mut_shape() = Shape({bbox_pred_blob_desc->shape().At(0), post_nms_top_n, 4});
  rois_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());

  // scores
  BlobDesc* scores_blob_desc = GetBlobDesc4BnInOp("scores");
  scores_blob_desc->mut_shape() = Shape({cls_prob_blob_desc->shape().At(0), post_nms_top_n, 1});
  rois_blob_desc->set_data_type(cls_prob_blob_desc->data_type());

  // fg_prob
  BlobDesc* fg_prob_blob_desc = GetBlobDesc4BnInOp("fg_scores");
  fg_prob_blob_desc->mut_shape() =
      Shape({cls_prob_blob_desc->shape().At(0),
             cls_prob_blob_desc->shape().Count(1, 3) * num_of_anchors, 1});
  fg_prob_blob_desc->set_data_type(cls_prob_blob_desc->data_type());

  // proposals
  *GetBlobDesc4BnInOp("proposals") = *bbox_pred_blob_desc;
}

REGISTER_OP(OperatorConf::kProposalConf, ProposalOp);

}  // namespace oneflow
