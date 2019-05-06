#include "oneflow/core/operator/parallel_cast_facade_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"

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

void ParallelCastFacadeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ParallelCastFacadeOpConf& conf = op_conf().parallel_cast_facade_conf();
  const SbpParallel& out_sbp_parallel = conf.out_sbp_parallel();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  if (out_sbp_parallel.has_partial_sum_parallel() || out_sbp_parallel.has_broadcast_parallel()) {
    out->mut_shape() = Shape(conf.logical_blob_shape());
  } else if (out_sbp_parallel.has_split_parallel()) {
    const int64_t axis = out_sbp_parallel.split_parallel().axis();
    CHECK_GE(axis, 0);
    CHECK_LT(axis, conf.logical_blob_shape().dim_size());
    BalancedSplitter splitter(conf.logical_blob_shape().dim(axis), parallel_ctx->parallel_num());
    out->mut_shape() = Shape(conf.logical_blob_shape());
    out->mut_shape().Set(axis, splitter.At(parallel_ctx->parallel_id()).size());
  } else {
    UNIMPLEMENTED();
  }
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
