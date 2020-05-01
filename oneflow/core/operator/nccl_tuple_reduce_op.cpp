#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class NcclTupleReduceOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleReduceOp);
  NcclTupleReduceOp() = default;
  ~NcclTupleReduceOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
  LogicalNode* NewProperLogicalNode() const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

void NcclTupleReduceOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_tuple_reduce_conf());
  EnrollRepeatedInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& NcclTupleReduceOp::GetCustomizedConf() const {
  return op_conf().nccl_tuple_reduce_conf();
}

Maybe<void> NcclTupleReduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NcclTupleReduceOpConf& conf = op_conf().nccl_tuple_reduce_conf();
  const int64_t num_blob = conf.in_size();
  CHECK_GE_OR_RETURN(num_blob, 1);
  CHECK_EQ_OR_RETURN(conf.out_size(), num_blob);
  CHECK_EQ_OR_RETURN(conf.root_size(), num_blob);
  FOR_RANGE(int32_t, i, 0, num_blob) {
    BlobDesc* out_i = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    if (out_i != nullptr) { *out_i = *in_i; }
  }
  return Maybe<void>::Ok();
}

Maybe<void> NcclTupleReduceOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& ibn : input_bns()) {
    CHECK_EQ_OR_RETURN(BatchAxis4BnInOp(ibn)->has_value(), false);
  }
  for (const auto& obn : output_bns()) { BatchAxis4BnInOp(obn)->clear_value(); }
  return Maybe<void>::Ok();
}

Maybe<void> NcclTupleReduceOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    CHECK_OR_RETURN(JUST(SbpInferHint4Ibn(ibn))->sbp_parallel().has_partial_sum_parallel());
  }
  SbpSignatureBuilder().PartialSum(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
  return Maybe<void>::Ok();
}

LogicalNode* NcclTupleReduceOp::NewProperLogicalNode() const {
  return new NcclTupleReduceLogicalNode();
}

void NcclTupleReduceOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  *kernel_conf->mutable_nccl_tuple_reduce_conf()->mutable_parallel_ctx() = *parallel_ctx;
}

REGISTER_OP(OperatorConf::kNcclTupleReduceConf, NcclTupleReduceOp);

}  // namespace oneflow
