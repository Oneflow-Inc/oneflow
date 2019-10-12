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

void NcclTupleBroadcastOp::InitFromOpConf() {
  CHECK(op_conf().has_nccl_tuple_broadcast_conf());
  if (op_conf().nccl_tuple_broadcast_conf().has_tick()) { EnrollInputBn("tick"); }
  EnrollRepeatedInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& NcclTupleBroadcastOp::GetCustomizedConf() const {
  return op_conf().nccl_tuple_broadcast_conf();
}

Maybe<void> NcclTupleBroadcastOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const NcclTupleBroadcastOpConf& conf = op_conf().nccl_tuple_broadcast_conf();
  const int64_t num_blob = conf.in_size();
  OF_CHECK_GE(num_blob, 1);
  OF_CHECK_EQ(conf.out_size(), num_blob);
  OF_CHECK_EQ(conf.data_type_size(), num_blob);
  OF_CHECK_EQ(conf.shape_size(), num_blob);
  OF_CHECK_EQ(conf.root_size(), num_blob);
  FOR_RANGE(int32_t, i, 0, num_blob) {
    const int64_t root = conf.root(i);
    OF_CHECK_LT(root, parallel_ctx->parallel_num());
    const Shape shape(conf.shape(i));
    const DataType data_type = conf.data_type(i);
    if (parallel_ctx->parallel_id() == root) {
      const BlobDesc* in_i = GetBlobDesc4BnInOp(GenRepeatedBn("in", i));
      OF_CHECK_EQ(in_i->data_type(), data_type);
      OF_CHECK_EQ(in_i->shape(), shape);
    }
    BlobDesc* out_i = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    out_i->mut_shape() = shape;
    out_i->set_data_type(data_type);
  }
  return Maybe<void>::Ok();
}

Maybe<void> NcclTupleBroadcastOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  for (const auto& ibn : input_bns()) {
    if (ibn == "tick") { continue; }
    OF_CHECK_EQ(BatchAxis4BnInOp(ibn)->has_value(), false);
  }
  for (const auto& obn : output_bns()) { BatchAxis4BnInOp(obn)->clear_value(); }
  return Maybe<void>::Ok();
}

Maybe<void> NcclTupleBroadcastOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  for (const auto& ibn : input_bns()) {
    if (ibn == "tick") { continue; }
    OF_CHECK(JUST(SbpInferHint4Ibn(ibn))->sbp_parallel().has_broadcast_parallel()
             || JUST(SbpInferHint4Ibn(ibn))->parallel_desc().parallel_num() == 1);
  }
  SbpSignatureBuilder().Broadcast(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
  return Maybe<void>::Ok();
}

LogicalNode* NcclTupleBroadcastOp::NewProperLogicalNode() const {
  return new NcclTupleBroadcastLogicalNode();
}

void NcclTupleBroadcastOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  *kernel_conf->mutable_nccl_tuple_broadcast_conf()->mutable_parallel_ctx() = *parallel_ctx;
}

REGISTER_OP(OperatorConf::kNcclTupleBroadcastConf, NcclTupleBroadcastOp);

}  // namespace oneflow
