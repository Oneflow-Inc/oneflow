#include "oneflow/core/operator/fpn_collect_op.h"

namespace oneflow {

void FpnCollectOp::InitFromOpConf() {
  CHECK(op_conf().has_relu_conf());
  EnrollRepeatedInputBn("rpn_rois_fpn");
  EnrollRepeatedInputBn("rpn_roi_probs_fpn");
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
  CHECK_GE(level <= input_bns().size()/2);
  // rpn_rois_fpn_i : (N, R, 5)
  // rpn_rois_probs_fpn_i : (N, R)
  int32_t post_nms_topn = conf.post_nms_top_n();
  BlobDesc* input_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(0));
  int32_t N = input_blob_desc->shape().At(0);
  int32_t R = input_blob_desc->shape().At(1);
  for (size_t i = 1; i < input_bns().size(); i++) {
    BlobDesc* input_blob_desc = GetBlobDesc4BnInOp(input_bns().Get(i))
    CHECK_EQ(roi_blob_desc->shape().At(0), N);
    CHECK_EQ(roi_blob_desc->shape().At(1), R);
  }
  // index (N * R) int32
  BlobDesc* index_blob_desc = GetBlobDesc4BnInOp("index");
  index_blob_desc->mut_shape() = Shape({N * R});
  index_blob_desc->set_data_type(DataType::kInt32);
  // roi_inputs (N, R, 5)
  BlobDesc* roi_inputs_blob_desc = GetBlobDesc4BnInOp("roi_inputs");
  roi_inputs_blob_desc->mut_shape() = Shape({N, R, 5});
  roi_inputs_blob_desc->set_data_type(input_blob_desc->data_type());
  // score_inputs (N, R)
  BlobDesc* score_inputs_blob_desc = GetBlobDesc4BnInOp("score_inputs");
  score_inputs_blob_desc->mut_shape() = Shape({N, R});
  score_inputs_blob_desc->set_data_type(input_blob_desc->data_type());
  // out (post_nms_topn ,5)
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({post_nms_topn, 5});
  out_blob_desc->set_data_type(input_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kFpnCollectConf, FpnCollectOp);

}  // namespace oneflow
