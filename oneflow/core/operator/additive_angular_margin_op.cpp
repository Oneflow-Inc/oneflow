#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AdditiveAngularMarginOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginOp);
  AdditiveAngularMarginOp() = default;
  ~AdditiveAngularMarginOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_additive_angular_margin_conf());
    EnrollInputBn("in");
    EnrollInputBn("label", false);
    EnrollOutputBn("sin_theta_data", false);
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().additive_angular_margin_conf();
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const override {
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
    const BlobDesc* label = GetBlobDesc4BnInOp("label");
    CHECK_GT_OR_RETURN(label->shape().NumAxes(), 0);
    CHECK_OR_RETURN(IsIntegralDataType(label->data_type()));
    CHECK_EQ_OR_RETURN(label->shape().At(0), in->shape().At(0));

    BlobDesc* sin_theta_data = GetBlobDesc4BnInOp("sin_theta_data");
    sin_theta_data->set_data_type(in->data_type());
    sin_theta_data->mut_shape() = label->shape();

    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");

    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder()
        .Split("label", 0)
        .Split("sin_theta_data", 0)
        .Split("in", 0)
        .Split("out", 0)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());

    return Maybe<void>::Ok();
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("sin_theta_data") = *BatchAxis4BnInOp("in");
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kAdditiveAngularMarginConf, AdditiveAngularMarginOp);

}  // namespace oneflow
