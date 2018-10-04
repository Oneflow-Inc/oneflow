#include "oneflow/core/operator/proposal_op.h"
#include "oneflow/core/common/protobuf.h"

namespace oneflow {

void ProposalOp::InitFromOpConf() {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_proposal_conf());
  EnrollInputBn("class_prob", false);
  EnrollInputBn("bbox_pred", false);
  EnrollOutputBn("rois", false);
  EnrollOutputBn("roi_probs", false);
  EnrollConstBufBn("anchors");
  EnrollDataTmpBn("proposals");
  EnrollDataTmpBn("pre_nms_slice");
  EnrollDataTmpBn("post_nms_slice");
}

const PbMessage& ProposalOp::GetCustomizedConf() const { return op_conf().proposal_conf(); }

void ProposalOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const BlobDesc* cls_prob_blob_desc = GetBlobDesc4BnInOp("class_prob");
  const BlobDesc* bbox_pred_blob_desc = GetBlobDesc4BnInOp("bbox_pred");
  const auto* anchor_generator_conf =
      GetMsgPtrFromPbMessage<AnchorGeneratorConf>(GetCustomizedConf(), "anchors_generator_conf");
  const auto& anchor_scales =
      GetPbRfFromPbMessage<int32_t>(*anchor_generator_conf, "anchor_scales");
  const auto& aspect_ratios = GetPbRfFromPbMessage<float>(*anchor_generator_conf, "aspect_ratios");
  const int32_t num_anchors = anchor_scales.size() * aspect_ratios.size();
  const int32_t fm_stride =
      GetValFromPbMessage<int32_t>(*anchor_generator_conf, "feature_map_stride");
  for (int32_t scale : anchor_scales) {
    CHECK_GE(scale, fm_stride);
    CHECK_EQ(scale % fm_stride, 0);
  }
  const int32_t pre_nms_top_n = GetValFromCustomizedConf<int32_t>("pre_nms_top_n");
  const int32_t post_nms_top_n = GetValFromCustomizedConf<int32_t>("post_nms_top_n");
  CHECK_GT(post_nms_top_n, 0);
  CHECK(pre_nms_top_n == -1 || pre_nms_top_n > post_nms_top_n);
  // in: class_prob (n, h, w, a)
  // in: bbox_pred (n, h, w, a * 4)
  FOR_RANGE(int32_t, i, 0, 3) {
    CHECK_EQ(cls_prob_blob_desc->shape().At(i), bbox_pred_blob_desc->shape().At(i));
  }
  CHECK_EQ(cls_prob_blob_desc->shape().At(3), num_anchors);
  CHECK_EQ(bbox_pred_blob_desc->shape().At(3), cls_prob_blob_desc->shape().At(3) * 4);
  const int64_t num_images = cls_prob_blob_desc->shape().At(0);
  const int64_t feature_map_h = cls_prob_blob_desc->shape().At(1);
  const int64_t feature_map_w = cls_prob_blob_desc->shape().At(2);
  // datatmp: proposals (h, w, a, 4)
  BlobDesc* proposal_blob_desc = GetBlobDesc4BnInOp("proposals");
  proposal_blob_desc->mut_shape() = Shape({feature_map_h, feature_map_w, num_anchors, 4});
  proposal_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());
  // const buf: anchors (h, w, a, 4)
  *GetBlobDesc4BnInOp("anchors") = *proposal_blob_desc;
  // datatmp: pre_nms_slice (h * w * a)
  BlobDesc* pre_nms_slice_blob_desc = GetBlobDesc4BnInOp("pre_nms_slice");
  pre_nms_slice_blob_desc->mut_shape() = Shape({feature_map_h * feature_map_w * num_anchors});
  pre_nms_slice_blob_desc->set_data_type(DataType::kInt32);
  // datatmp: post_nms_slice
  BlobDesc* post_nms_slice_blob_desc = GetBlobDesc4BnInOp("post_nms_slice");
  post_nms_slice_blob_desc->mut_shape() = Shape({post_nms_top_n});
  post_nms_slice_blob_desc->set_data_type(DataType::kInt32);
  // out: rois (n, r, 5)
  BlobDesc* rois_blob_desc = GetBlobDesc4BnInOp("rois");
  rois_blob_desc->mut_shape() = Shape({num_images, post_nms_top_n, 5});
  rois_blob_desc->set_data_type(bbox_pred_blob_desc->data_type());
  rois_blob_desc->set_has_data_id_field(bbox_pred_blob_desc->has_data_id_field());
  // out: roi_probs (n, r)
  BlobDesc* roi_probs_blob_desc = GetBlobDesc4BnInOp("roi_probs");
  roi_probs_blob_desc->mut_shape() = Shape({num_images, post_nms_top_n});
  roi_probs_blob_desc->set_data_type(cls_prob_blob_desc->data_type());
  roi_probs_blob_desc->set_has_data_id_field(cls_prob_blob_desc->has_data_id_field());
}

REGISTER_OP(OperatorConf::kProposalConf, ProposalOp);

}  // namespace oneflow
