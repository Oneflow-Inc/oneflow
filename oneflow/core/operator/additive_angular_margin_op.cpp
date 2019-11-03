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
    const BlobDesc* label = GetBlobDesc4BnInOp("label");
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

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    kernel_conf->mutable_additive_angular_margin_conf()->set_label_type(
        GetBlobDesc4BnInOp("label")->data_type());
  }
};

class AdditiveAngularMarginMs1Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AdditiveAngularMarginMs1Op);
  AdditiveAngularMarginMs1Op() = default;
  ~AdditiveAngularMarginMs1Op() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_additive_angular_margin_ms1_conf());
    EnrollInputBn("in");
    EnrollInputBn("label", false);
    EnrollOutputBn("sin_theta_data", false);
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().additive_angular_margin_ms1_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const override {
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    const BlobDesc* label = GetBlobDesc4BnInOp("label");
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
        .Broadcast("label")
        .PartialSum("sin_theta_data")
        .Split("in", 1)
        .Split("out", 1)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());

    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    const int64_t depth = op_conf().additive_angular_margin_ms1_conf().depth();
    CHECK_GE(depth, parallel_ctx->parallel_num());
    BalancedSplitter bs(depth, parallel_ctx->parallel_num());
    kernel_conf->mutable_additive_angular_margin_conf()->set_lower_bound(
        bs.At(parallel_ctx->parallel_id()).begin());
    kernel_conf->mutable_additive_angular_margin_conf()->set_label_type(
        GetBlobDesc4BnInOp("label")->data_type());
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("sin_theta_data") = *BatchAxis4BnInOp("in");
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
    return Maybe<void>::Ok();
  }
};

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

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    kernel_conf->mutable_additive_angular_margin_grad_conf()->set_label_type(
        GetBlobDesc4BnInOp("label")->data_type());
  }
};

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
    kernel_conf->mutable_additive_angular_margin_grad_conf()->set_label_type(
        GetBlobDesc4BnInOp("label")->data_type());
  }

  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    *BatchAxis4BnInOp("dx") = *BatchAxis4BnInOp("dy");
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kAdditiveAngularMarginConf, AdditiveAngularMarginOp);
REGISTER_OP(OperatorConf::kAdditiveAngularMarginMs1Conf, AdditiveAngularMarginMs1Op);
REGISTER_OP(OperatorConf::kAdditiveAngularMarginGradConf, AdditiveAngularMarginGradOp);
REGISTER_OP(OperatorConf::kAdditiveAngularMarginMs1GradConf, AdditiveAngularMarginMs1GradOp);

}  // namespace oneflow
