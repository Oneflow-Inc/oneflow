#include "oneflow/core/operator/scalar_op_base.h"

namespace oneflow {

class ScalarDivByTensorOp final : public ScalarOpBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarDivByTensorOp);
  ScalarDivByTensorOp() = default;
  ~ScalarDivByTensorOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> VirtualGetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

const PbMessage& ScalarDivByTensorOp::GetCustomizedConf() const {
  return op_conf().scalar_div_by_tensor_conf();
}

Maybe<void> ScalarDivByTensorOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().PartialSum("in").Broadcast("scalar").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kScalarDivByTensorConf, ScalarDivByTensorOp);

}  // namespace oneflow
