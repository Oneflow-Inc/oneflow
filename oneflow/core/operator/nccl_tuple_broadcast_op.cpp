#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class NcclTupleBroadcastOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclTupleBroadcastOp);
  NcclTupleBroadcastOp() = default;
  ~NcclTupleBroadcastOp() override = default;

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

void NcclTupleBroadcastOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_tuple_broadcast_conf());
  if (op_conf().nccl_tuple_broadcast_conf().has_tick()) { EnrollInputBn("tick"); }
  EnrollRepeatedInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& NcclTupleBroadcastOp::GetCustomizedConf() const {
  return op_conf().nccl_tuple_broadcast_conf();
}

void NcclTupleBroadcastOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NcclTupleBroadcastOpConf& conf = op_conf().nccl_tuple_broadcast_conf();
  const int64_t num_blob = conf.in_size();
  CHECK_GE(num_blob, 1);
  CHECK_EQ(conf.out_size(), num_blob);
  CHECK_EQ(conf.data_type_size(), num_blob);
  CHECK_EQ(conf.shape_size(), num_blob);
  CHECK_EQ(conf.root_size(), num_blob);
  FOR_RANGE(int32_t, i, 0, num_blob) {
    const int64_t root = conf.root(i);
    CHECK_LT(root, parallel_ctx->parallel_num());
    const Shape shape(conf.shape(i));
    const DataType data_type = conf.data_type(i);
    if (parallel_ctx->parallel_id() == root) {
      const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
      CHECK_EQ(in_i->data_type(), data_type);
      CHECK_EQ(in_i->shape(), shape);
    }
    BlobDesc* out_i = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    out_i->mut_shape() = shape;
    out_i->set_data_type(data_type);
  }
}

void NcclTupleBroadcastOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  for (const auto& ibn : input_bns()) {
    if (ibn == "tick") { continue; }
    CHECK_EQ(*HasBatchDim4BnInOp(ibn), false);
  }
  for (const auto& obn : output_bns()) { *HasBatchDim4BnInOp(obn) = false; }
}

void NcclTupleBroadcastOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    if (ibn == "tick") { continue; }
    CHECK(SbpInferHint4Ibn(ibn).sbp_parallel().has_broadcast_parallel()
          || SbpInferHint4Ibn(ibn).parallel_desc().parallel_num() == 1);
  }
  SbpSignatureBuilder().Broadcast(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
}

LogicalNode* NcclTupleBroadcastOp::NewProperLogicalNode() const {
  return new NcclTupleBroadcastLogicalNode();
}

REGISTER_OP(OperatorConf::kNcclTupleBroadcastConf, NcclTupleBroadcastOp);

}  // namespace oneflow
