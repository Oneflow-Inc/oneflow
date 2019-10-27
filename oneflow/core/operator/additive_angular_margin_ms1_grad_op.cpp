#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/operator.h"
namespace oneflow {

class AdditiveAngularMarginMs1GradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginMs1GradOp);
  AdditiveAngularMarginMs1GradOp() = default;
  ~AdditiveAngularMarginMs1GradOp() override = default;
  void InitFromOpConf() override {
    CHECK(op_conf().has_additive_angular_margin_ms1_grad_conf());
    EnrollInputBn("dy");
    EnrollInputBn("label", false);
    EnrollInputBn("sin_theta_data", false);
    EnrollOutputBn("dx");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().additive_angular_margin_ms1_grad_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
    const BlobDesc* label = GetBlobDesc4BnInOp("label");
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
        .Broadcast("label")
        .Broadcast("sin_theta_data")
        .Split("dy", 1)
        .Split("dx", 1)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());

    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    const int64_t depth = op_conf().additive_angular_margin_ms1_grad_conf().depth();
    CHECK_GE(depth, parallel_ctx->parallel_num());
    BalancedSplitter bs(depth, parallel_ctx->parallel_num());
    kernel_conf->mutable_additive_angular_margin_grad_conf()->set_lower_bound(
        bs.At(parallel_ctx->parallel_id()).begin());
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("dx") = *BatchAxis4BnInOp("dy");
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kAdditiveAngularMarginMs1GradConf, AdditiveAngularMarginMs1GradOp);

}  // namespace oneflow
