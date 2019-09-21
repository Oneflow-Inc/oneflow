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

};  // namespace oneflow

REGISTER_OP(OperatorConf::kSmoothL1Conf, SmoothL1Op);

}  // namespace oneflow
