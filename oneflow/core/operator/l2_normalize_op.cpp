#include "oneflow/core/operator/l2_normalize_op.h"

namespace oneflow {

void L2NormalizeOp::InitFromOpConf() {
  CHECK(op_conf().has_l2_normalize_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollTmpBn("square_x_sum");
}

const PbMessage& L2NormalizeOp::GetCustomizedConf() const { return op_conf().l2_normalize_conf(); }

void L2NormalizeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const L2NormalizeOpConf& conf = op_conf().l2_normalize_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int32_t axis_num = in_blob_desc->shape().NumAxes();
  int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + axis_num;
  CHECK_GE(axis, 0);
  CHECK_LT(axis, axis_num);
  CHECK_GT(conf.epsilon(), 0);
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  BlobDesc* square_x_sum_blob_desc = GetBlobDesc4BnInOp("square_x_sum");
  *square_x_sum_blob_desc = *in_blob_desc;
  square_x_sum_blob_desc->mut_shape().Set(axis, 1);
}

REGISTER_OP(OperatorConf::kL2NormalizeConf, L2NormalizeOp);

}  // namespace oneflow
