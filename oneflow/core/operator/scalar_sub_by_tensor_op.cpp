#include "oneflow/core/operator/scalar_op_base.h"

namespace oneflow {

class ScalarSubByTensorOp final : public ScalarOpBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarSubByTensorOp);
  ScalarSubByTensorOp() = default;
  ~ScalarSubByTensorOp() override = default;

 private:
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> VirtualGetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

const PbMessage& ScalarSubByTensorOp::GetCustomizedConf() const {
  return op_conf().scalar_sub_by_tensor_conf();
}

Maybe<void> ScalarSubByTensorOp::VirtualGetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().PartialSum("in").PartialSum("scalar").PartialSum("out").Build(
      sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kScalarSubByTensorConf, ScalarSubByTensorOp);

}  // namespace oneflow
