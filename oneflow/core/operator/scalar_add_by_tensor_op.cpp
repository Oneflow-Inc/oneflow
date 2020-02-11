#include "oneflow/core/operator/scalar_op_base.h"

namespace oneflow {

class ScalarAddByTensorOp final : public ScalarOpBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarAddByTensorOp);
  ScalarAddByTensorOp() = default;
  ~ScalarAddByTensorOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> VirtualGetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

const PbMessage& ScalarAddByTensorOp::GetCustomizedConf() const {
  return op_conf().scalar_add_by_tensor_conf();
}

Maybe<void> ScalarAddByTensorOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().PartialSum("in").PartialSum("scalar").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kScalarAddByTensorConf, ScalarAddByTensorOp);

}  // namespace oneflow
