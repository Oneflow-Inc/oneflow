#include "oneflow/core/operator/every_nth_op.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void EveryNthOp::InitFromOpConf() {
  CHECK(op_conf().has_every_nth_conf());
  const EveryNthOpConf& conf = op_conf().every_nth_conf();
  CHECK_GE(conf.n(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void EveryNthOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  const int64_t n = op_conf().every_nth_conf().n();
  const Shape* in_shape = GetTimeShape4BnInOp("in");
  std::vector<int64_t> dim_vec;
  CHECK_GE(in_shape->NumAxes(), 1);
  CHECK_GE(n, 1);
  if (in_shape->dim_vec().back() % n == 0) {
    dim_vec.insert(dim_vec.end(), in_shape->dim_vec().begin(), in_shape->dim_vec().cend() - 1);
    if (in_shape->dim_vec().back() != n) { dim_vec.push_back(in_shape->dim_vec().back() / n); }
  } else {
    dim_vec.push_back(in_shape->elem_cnt() / n);
  }
  *time_shape = Shape(dim_vec);
}

const PbMessage& EveryNthOp::GetCustomizedConf() const { return op_conf().every_nth_conf(); }

void EveryNthOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

LogicalNode* EveryNthOp::NewProperLogicalNode() { return new EveryNthLogicalNode(); }

void EveryNthOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(MakeIdentitySbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kEveryNthConf, EveryNthOp);

}  // namespace oneflow
