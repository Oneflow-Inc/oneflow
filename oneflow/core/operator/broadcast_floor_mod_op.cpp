#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastFloorModOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastFloorModOp);
  BroadcastFloorModOp() = default;
  ~BroadcastFloorModOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> VirtualGetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

const PbMessage& BroadcastFloorModOp::GetCustomizedConf() const {
  return op_conf().broadcast_floor_mod_conf();
}

Maybe<void> BroadcastFloorModOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().PartialSum("a").Broadcast("b").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBroadcastFloorModConf, BroadcastFloorModOp);

}  // namespace oneflow
