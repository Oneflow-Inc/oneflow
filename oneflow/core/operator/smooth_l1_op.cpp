#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SmoothL1Op final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1Op);
  SmoothL1Op() = default;
  ~SmoothL1Op() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_smooth_l1_conf());
    EnrollInputBn("prediction");
    EnrollInputBn("label", false);
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().smooth_l1_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* prediction = GetBlobDesc4BnInOp("prediction");
    const BlobDesc* label = GetBlobDesc4BnInOp("label");
    CHECK_EQ_OR_RETURN(prediction->shape(), label->shape());
    CHECK_EQ_OR_RETURN(prediction->data_type(), label->data_type());
    CHECK_GE_OR_RETURN(op_conf().smooth_l1_conf().beta(), 0);

    // out
    *GetBlobDesc4BnInOp("out") = *prediction;
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

REGISTER_OP(OperatorConf::kSmoothL1Conf, SmoothL1Op);

class SmoothL1GradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SmoothL1GradOp);
  SmoothL1GradOp() = default;
  ~SmoothL1GradOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_smooth_l1_grad_conf());
    EnrollInputBn("x");
    EnrollInputBn("dy");
    EnrollInputBn("label", false);
    EnrollOutputBn("dx");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().smooth_l1_grad_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* x = GetBlobDesc4BnInOp("x");
    const BlobDesc* dy = GetBlobDesc4BnInOp("dy");
    const BlobDesc* label = GetBlobDesc4BnInOp("label");
    CHECK_EQ_OR_RETURN(x->shape(), dy->shape());
    CHECK_EQ_OR_RETURN(x->data_type(), dy->data_type());
    CHECK_EQ_OR_RETURN(x->shape(), label->shape());
    CHECK_EQ_OR_RETURN(x->data_type(), label->data_type());
    CHECK_GE_OR_RETURN(op_conf().smooth_l1_grad_conf().beta(), 0);

    // out
    *GetBlobDesc4BnInOp("dx") = *dy;
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

REGISTER_OP(OperatorConf::kSmoothL1GradConf, SmoothL1GradOp);
}  // namespace oneflow
