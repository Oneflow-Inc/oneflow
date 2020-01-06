#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/operator/repeat_op.h"

namespace oneflow {

void RepeatOp::InitFromOpConf() {
  CHECK(op_conf().has_repeat_conf());
  const RepeatOpConf& conf = op_conf().repeat_conf();
  CHECK_GE(conf.repeat_num(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> RepeatOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  DimVector dim_vec(GetTimeShape4BnInOp("in")->dim_vec());
  dim_vec.push_back(op_conf().repeat_conf().repeat_num());
  *time_shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

const PbMessage& RepeatOp::GetCustomizedConf() const { return op_conf().repeat_conf(); }

Maybe<void> RepeatOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> RepeatOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return NaiveInferBatchAxis(BatchAxis4BnInOp);
}

Maybe<void> RepeatOp::InferSbpSignature(
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

LogicalNode* RepeatOp::NewProperLogicalNode() const { return new RepeatForwardLogicalNode(); }

REGISTER_OP(OperatorConf::kRepeatConf, RepeatOp);

}  // namespace oneflow
