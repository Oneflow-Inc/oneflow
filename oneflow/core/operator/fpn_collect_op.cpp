#include "oneflow/core/operator/fpn_collect_op.h"

namespace oneflow {

void FpnCollectOp::InitFromOpConf() {
  CHECK(op_conf().has_relu_conf());
  // rois:[r,5] probs:[r]
  for (int32_t i = 2; i <= 6; i++) {
    std::string roi_bn = "rpn_rois_fpn_" + std::to_string(i);
    std::string prob_bn = "rpn_roi_probs_fpn_" + std::to_string(i);
    EnrollInputBn(roi_bn);
    EnrollInputBn(prob_bn);
  }

  EnrollOutputBn("out");
  EnrollDataTmpBn("roi_inputs");
  EnrollDataTmpBn("index");
  EnrollDataTmpBn("score_inputs");
}

const PbMessage& FpnCollectOp::GetCustomizedConf() const { return op_conf().fpn_collect_conf(); }

void FpnCollectOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const FpnCollectOpConf& conf = op_conf().fpn_collect_conf();
  int32_t level = conf.level();
  int32_t topn_R = conf.post_nms_topn();
  int32_t R = 0;
  BlobDesc* roi_blob_desc_2 = GetBlobDesc4BnInOp("rpn_rois_fpn_2");
  BlobDesc* prob_blob_desc_2 = GetBlobDesc4BnInOp("rpn_rois_probs_fpn_2");
  int32_t N = roi_blob_desc_2->shape().At(0);
  // rpn_rois_fpn_i : (N,ri,4)
  // rpn_rois_probs_fpn_i : (N,ri)
  for (int32_t i = 2; i <= level; i++) {
    std::string roi_bn = "rpn_rois_fpn_" + std::to_string(i);
    BlobDesc* roi_blob_desc = GetBlobDesc4BnInOp(roi_bn);
    CHECK_EQ(roi_blob_desc->shape().At(0), N);
    R += roi_blob_desc->shape().At(1);
  }
  // index (N,R) int32
  BlobDesc* index_blob_desc = GetBlobDesc4BnInOp("index");
  index_blob_desc->mut_shape() = Shape({N, R});
  index_blob_desc->set_data_type(DataType::kInt32);
  // roi_inputs (N,R,4)
  BlobDesc* roi_inputs_blob_desc = GetBlobDesc4BnInOp("roi_inputs");
  roi_inputs_blob_desc->mut_shape() = Shape({N, R, 4});
  roi_inputs_blob_desc->set_data_type(roi_blob_desc_2->data_type());
  // score_inputs (R)
  BlobDesc* score_inputs_blob_desc = GetBlobDesc4BnInOp("score_inputs");
  score_inputs_blob_desc->mut_shape() = Shape({N, R});
  score_inputs_blob_desc->set_data_type(prob_blob_desc_2->data_type());
  // out (topR ,4)
  CHECK_GE(N * R, topn_R);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({topn_R, 5});
  out_blob_desc->set_data_type(roi_blob_desc_2->data_type());
}

REGISTER_OP(OperatorConf::kFpnCollectConf, FpnCollectOp);

}  // namespace oneflow
