#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class RegularGradientOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RegularGradientOp);
  RegularGradientOp() = default;
  ~RegularGradientOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void RegularGradientOp::InitFromOpConf() {
  CHECK(op_conf().has_regular_gradient_conf());
  EnrollInputBn("model", false);
  EnrollInputBn("model_diff", false);
  EnrollOutputBn("out", false)->set_mutable_inplace_ibn("model_diff");
}

const PbMessage& RegularGradientOp::GetCustomizedConf() const {
  return op_conf().regular_gradient_conf();
}

Maybe<void> RegularGradientOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model = GetBlobDesc4BnInOp("model");
  const BlobDesc* model_diff = GetBlobDesc4BnInOp("model_diff");
  CHECK_OR_RETURN(*model == *model_diff);
  *GetBlobDesc4BnInOp("out") = *model_diff;
  return Maybe<void>::Ok();
}

Maybe<void> RegularGradientOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  OF_CHECK(!BatchAxis4BnInOp("model")->has_value());
  OF_CHECK(!BatchAxis4BnInOp("model_diff")->has_value());
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> RegularGradientOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kRegularGradientConf, RegularGradientOp);

}  // namespace oneflow
