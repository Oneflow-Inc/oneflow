#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AdditiveAngularMarginGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginGradOp);
  AdditiveAngularMarginGradOp() = default;
  ~AdditiveAngularMarginGradOp() override = default;
  void InitFromOpConf() override {
    CHECK(op_conf().has_additive_angular_margin_grad_conf());
    EnrollInputBn("dy");
    EnrollInputBn("label", false);
    EnrollInputBn("sin_theta_data", false);
    EnrollOutputBn("dx");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().additive_angular_margin_grad_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
    CHECK_GT_OR_RETURN(dy->shape().NumAxes(), 0);
    const BlobDesc* label = GetBlobDesc4BnInOp("label");
    CHECK_GT_OR_RETURN(label->shape().NumAxes(), 0);
    CHECK_OR_RETURN(IsIntegralDataType(label->data_type()));
    CHECK_EQ_OR_RETURN(label->shape().At(0), dy->shape().At(0));

    const BlobDesc* sin_theta_data = GetBlobDesc4BnInOp("sin_theta_data");
    CHECK_EQ_OR_RETURN(sin_theta_data->shape().At(0), label->shape().At(0));

    *GetBlobDesc4BnInOp("dx") = *dy;
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("label", 0)
        .Split("dy", 0)
        .Split("sin_theta_data", 0)
        .Split("dx", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());

    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("dx") = *BatchAxis4BnInOp("dy");
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kAdditiveAngularMarginGradConf, AdditiveAngularMarginGradOp);

}  // namespace oneflow
