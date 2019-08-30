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

Maybe<void> EveryNthOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  const int64_t n = op_conf().every_nth_conf().n();
  const Shape* in_shape = GetTimeShape4BnInOp("in");
  std::vector<int64_t> dim_vec;
  CHECK_GE_OR_RETURN(in_shape->NumAxes(), 1);
  CHECK_GE_OR_RETURN(n, 1);
  if (in_shape->dim_vec().back() % n == 0) {
    dim_vec.insert(dim_vec.end(), in_shape->dim_vec().begin(), in_shape->dim_vec().cend() - 1);
    if (in_shape->dim_vec().back() != n) { dim_vec.push_back(in_shape->dim_vec().back() / n); }
  } else {
    dim_vec.push_back(in_shape->elem_cnt() / n);
  }
  *time_shape = Shape(dim_vec);
  return Maybe<void>::Ok();
}

const PbMessage& EveryNthOp::GetCustomizedConf() const { return op_conf().every_nth_conf(); }

Maybe<void> EveryNthOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

void EveryNthOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto bns = StdVec2PbRpf<std::string>({"in", "out"});
  SbpSignatureBuilder().Broadcast(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  const int64_t num_axes = LogicalBlobDesc4Ibn("in").shape().NumAxes();
  SbpSignatureBuilder().Split(bns, 0).MakeSplitSignatureListBuilder(num_axes).Build(sbp_sig_list);
}

LogicalNode* EveryNthOp::NewProperLogicalNode() const { return new EveryNthLogicalNode(); }

REGISTER_OP(OperatorConf::kEveryNthConf, EveryNthOp);

}  // namespace oneflow
