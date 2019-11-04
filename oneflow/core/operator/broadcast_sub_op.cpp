#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastSubOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastSubOp);
  BroadcastSubOp() = default;
  ~BroadcastSubOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> VirtualGetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

const PbMessage& BroadcastSubOp::GetCustomizedConf() const {
  return op_conf().broadcast_sub_conf();
}

Maybe<void> BroadcastSubOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().PartialSum("a").PartialSum("b").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBroadcastSubConf, BroadcastSubOp);

}  // namespace oneflow
