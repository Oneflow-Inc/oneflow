#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastMulOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastMulOp);
  BroadcastMulOp() = default;
  ~BroadcastMulOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> VirtualGetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

const PbMessage& BroadcastMulOp::GetCustomizedConf() const {
  return op_conf().broadcast_mul_conf();
}

Maybe<void> BroadcastMulOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Broadcast("a").PartialSum("b").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder().PartialSum("a").Broadcast("b").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBroadcastMulConf, BroadcastMulOp);

}  // namespace oneflow
