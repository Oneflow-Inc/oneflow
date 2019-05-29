#include "oneflow/core/operator/matcher_op.h"

namespace oneflow {

void MatcherOp::InitFromOpConf() {
  CHECK(op_conf().has_matcher_conf());
  EnrollInputBn("iou_matrix", false);
  EnrollInputBn("iou_matrix_shape", false);
  EnrollOutputBn("matched_indices", false);
}

const PbMessage& MatcherOp::GetCustomizedConf() const { return this->op_conf().matcher_conf(); }

// split Matcher into top_1 with keep_same_val and set_low_quality_matches
void MatcherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  // input: iou_matrix (N, G, M)
  const BlobDesc* iou_matrix = GetBlobDesc4BnInOp("iou_matrix");
  CHECK_EQ(iou_matrix->shape().NumAxes(), 3);
  const int64_t num_imgs = iou_matrix->shape().At(0);
  // input: iou_matrix_shape (N, 2)
  const BlobDesc* iou_matrix_shape = GetBlobDesc4BnInOp("iou_matrix_shape");
  CHECK_EQ(iou_matrix_shape->shape().NumAxes(), 2);
  CHECK_EQ(num_imgs, iou_matrix_shape->shape().At(0));
  // output: matched_indices (N, M)
  BlobDesc* matched_indices = GetBlobDesc4BnInOp("matched_indices");
  matched_indices->mut_shape() = Shape({num_imgs, iou_matrix->shape().At(1)});
  matched_indices->set_data_type(DataType::kInt32);
  matched_indices->set_has_dim1_valid_num_field(true);
}

REGISTER_OP(OperatorConf::kMatcherConf, MatcherOp);

}  // namespace oneflow
