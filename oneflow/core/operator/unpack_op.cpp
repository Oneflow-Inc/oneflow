#include "oneflow/core/operator/unpack_op.h"

namespace oneflow {

void UnpackOp::InitFromOpConf() {
  CHECK(op_conf().has_unpack_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> UnpackOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp(SoleIbn());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp(SoleObn());
  *out_blob_desc = *in_blob_desc;
  int32_t unpack_num = op_conf().unpack_conf().unpack_num();
  CHECK_EQ_OR_RETURN(0, in_blob_desc->shape().At(0) % unpack_num);
  out_blob_desc->mut_shape().Set(0, out_blob_desc->shape().At(0) / unpack_num);
  return Maybe<void>::Ok();
}

Maybe<void> UnpackOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  DimVector dim_vec(GetTimeShape4BnInOp("in")->dim_vec());
  int32_t unpack_num = op_conf().unpack_conf().unpack_num();
  dim_vec.push_back(unpack_num);
  *time_shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> UnpackOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return NaiveInferBatchAxis(BatchAxis4BnInOp);
}

Maybe<void> UnpackOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
  const SbpParallel sbp_parallel = JUST(SbpInferHint4Ibn("in"))->sbp_parallel();
  (*bn2sbp)["in"] = sbp_parallel;
  (*bn2sbp)["out"] = sbp_parallel;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kUnpackConf, UnpackOp);

}  // namespace oneflow
