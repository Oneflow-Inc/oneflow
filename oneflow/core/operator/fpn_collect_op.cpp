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
  EnrollDataTmpBn("score_inputs");
}

const PbMessage& FpnCollectOp::GetCustomizedConf() const { return op_conf().fpn_collect_conf(); }

void FpnCollectOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const FpnCollectOpConf& conf = op_conf().fpn_collect_conf();
  int32_t level = conf.level();
  int32_t N = conf.post_nms_topn();
  int32_t R = 0;
  for (int32_t i = 2; i <= level; i++) {
    std::string roi_bn = "rpn_rois_fpn_" + std::to_string(i);
    BlobDesc* roi_blob_desc = GetBlobDesc4BnInOp(roi_bn);
    R += roi_blob_desc->shape().At(0);
  }

  // roi_inputs
  BlobDesc* roi_inputs_blob_desc = GetBlobDesc4BnInOp("roi_inputs");
  roi_inputs_blob_desc->mut_shape() = Shape({R, 5});
  // score_inputs
  BlobDesc* score_inputs_blob_desc = GetBlobDesc4BnInOp("score_inputs");
  score_inputs_blob_desc->mut_shape() = Shape({R});
  // out
  CHECK_GE(R, N);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({N, 5});
}

REGISTER_OP(OperatorConf::kFpnCollectConf, FpnCollectOp);

}  // namespace oneflow
