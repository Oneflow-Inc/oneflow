#include "oneflow/core/operator/l2_normalize_op.h"

namespace oneflow {

void L2NormalizeOp::InitFromOpConf() {
  CHECK(op_conf().has_l2_normalize_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  // EnrollDataTmpBn("norm_data");
}

const PbMessage& L2NormalizeOp::GetCustomizedConf() const { return op_conf().l2_normalize_conf(); }

void L2NormalizeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const L2NormalizeOpConf& conf = op_conf().l2_normalize_conf();
  CHECK_GT(conf.axis(), 0);
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  // // tmp: norm_data_blob_desc (n * h * w)
  // BlobDesc* norm_data_blob_desc = GetBlobDesc4BnInOp("norm_data");
  // norm_data_blob_desc->mut_shape() = Shape(
  //     {in_blob_desc->shape().At(0) * in_blob_desc->shape().At(1) * in_blob_desc->shape().At(2)});
  // norm_data_blob_desc->set_data_type(in_blob_desc->data_type());
}

REGISTER_OP(OperatorConf::kL2NormalizeConf, L2NormalizeOp);

}  // namespace oneflow
