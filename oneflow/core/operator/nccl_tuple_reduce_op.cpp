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

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void InferSbpSignature(SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
                         const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
                         std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
                         const ParallelDesc& parallel_desc) const override;
  LogicalNode* NewProperLogicalNode() const override;
};

void NcclTupleReduceOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_tuple_reduce_conf());
  EnrollRepeatedInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& NcclTupleReduceOp::GetCustomizedConf() const {
  return op_conf().nccl_tuple_reduce_conf();
}

void NcclTupleReduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NcclTupleReduceOpConf& conf = op_conf().nccl_tuple_reduce_conf();
  const int64_t num_blob = conf.in_size();
  CHECK_GE(num_blob, 1);
  CHECK_EQ(conf.out_size(), num_blob);
  CHECK_EQ(conf.root_size(), num_blob);
  FOR_RANGE(int32_t, i, 0, num_blob) {
    BlobDesc* out_i = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
    if (out_i != nullptr) { *out_i = *in_i; }
  }
}

void NcclTupleReduceOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  for (const auto& ibn : input_bns()) { CHECK_EQ(*HasBatchDim4BnInOp(ibn), false); }
  for (const auto& obn : output_bns()) { *HasBatchDim4BnInOp(obn) = false; }
}

void NcclTupleReduceOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    CHECK(SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel());
  }
  SbpSignatureBuilder().PartialSum(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
}

LogicalNode* NcclTupleReduceOp::NewProperLogicalNode() const {
  return new NcclTupleReduceLogicalNode();
}

REGISTER_OP(OperatorConf::kNcclTupleReduceConf, NcclTupleReduceOp);

}  // namespace oneflow
