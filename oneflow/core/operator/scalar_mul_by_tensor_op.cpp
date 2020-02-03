#include "oneflow/core/operator/scalar_op_base.h"

namespace oneflow {

class ScalarMulByTensorOp final : public ScalarOpBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarMulByTensorOp);
  ScalarMulByTensorOp() = default;
  ~ScalarMulByTensorOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> VirtualGetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

const PbMessage& ScalarMulByTensorOp::GetCustomizedConf() const {
  return op_conf().scalar_mul_by_tensor_conf();
}

Maybe<void> ScalarMulByTensorOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().PartialSum("in").Broadcast("scalar").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder().Broadcast("in").PartialSum("scalar").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kScalarMulByTensorConf, ScalarMulByTensorOp);

}  // namespace oneflow
