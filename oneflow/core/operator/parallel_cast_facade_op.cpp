#include "oneflow/core/operator/parallel_cast_facade_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ParallelCastFacadeOp::InitFromOpConf() {
  CHECK(op_conf().has_parallel_cast_facade_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& ParallelCastFacadeOp::GetCustomizedConf() const {
  return op_conf().parallel_cast_facade_conf();
}

LogicalNode* ParallelCastFacadeOp::NewProperLogicalNode() const {
  return new ParallelCastFacadeLogicalNode();
}

void ParallelCastFacadeOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = *HasBatchDim4BnInOp("in");
}

void ParallelCastFacadeOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  const ParallelCastFacadeOpConf& conf = op_conf().parallel_cast_facade_conf();
  (*sbp_signature->mutable_bn_in_op2sbp_parallel())["in"] = conf.in_sbp_parallel();
  (*sbp_signature->mutable_bn_in_op2sbp_parallel())["out"] = conf.out_sbp_parallel();
}

REGISTER_OP(OperatorConf::kParallelCastFacadeConf, ParallelCastFacadeOp);

}  // namespace oneflow
