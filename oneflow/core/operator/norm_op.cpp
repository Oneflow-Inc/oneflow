#include "oneflow/core/operator/norm_op.h"

namespace oneflow {

void NormOp::InitFromOpConf() {
  CHECK(op_conf().has_norm_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("abs_tmp");
  EnrollDataTmpBn("sum_tmp");
}

const PbMessage& NormOp::GetCustomizedConf() const { return op_conf().norm_conf(); }

void NormOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  const NormOpConf& conf = op_conf().norm_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  int32_t axis_num = in_blob_desc->shape().NumAxes();
  int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + axis_num;
  CHECK_GE(axis, 0);
  CHECK_LT(axis, axis_num);
  CHECK_GT(conf.epsilon(), 0);
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  std::vector<int64_t> out_shape;
  for (size_t i = 0; i < axis; ++i) { out_shape.push_back(in_blob_desc->shape().At(i)); }
  out_blob_desc->mut_shape() = Shape(out_shape);
  *GetBlobDesc4BnInOp("abs_tmp") = *in_blob_desc;
  *GetBlobDesc4BnInOp("sum_tmp") = *in_blob_desc;
  int32_t buf_size = in_blob_desc->shape().elem_cnt() / out_blob_desc->shape().elem_cnt();
  GetBlobDesc4BnInOp("sum_tmp")->mut_shape() = Shape({buf_size});
}

REGISTER_OP(OperatorConf::kNormConf, NormOp);

}  // namespace oneflow
