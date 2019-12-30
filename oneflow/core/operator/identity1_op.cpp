#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class Identity1Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Identity1Op);
  Identity1Op() = default;
  ~Identity1Op() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void Identity1Op::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

Maybe<void> Identity1Op::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

const PbMessage& Identity1Op::GetCustomizedConf() const { return op_conf().identity1_conf(); }

Maybe<void> Identity1Op::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return NaiveInferBatchAxis(BatchAxis4BnInOp);
}

Maybe<void> Identity1Op::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kIdentity1Conf, Identity1Op);

}  // namespace oneflow
