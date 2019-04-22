#include "oneflow/core/operator/all_reduce_facade_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/sbp_signature_rule.h"

namespace oneflow {

void AllReduceFacadeOp::InitFromOpConf() {
  CHECK(op_conf().has_all_reduce_facade_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& AllReduceFacadeOp::GetCustomizedConf() const {
  return op_conf().all_reduce_facade_conf();
}

void AllReduceFacadeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  *out_blob = *in_blob;
}

void AllReduceFacadeOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(MakeP2BSignatureRule(this));
}

LogicalNode* AllReduceFacadeOp::NewProperLogicalNode() const {
  return new AllReduceFacadeLogicalNode();
}

REGISTER_OP(OperatorConf::kAllReduceFacadeConf, AllReduceFacadeOp);

}  // namespace oneflow
