#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ConstantLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstantLikeOp);
  ConstantLikeOp() = default;
  ~ConstantLikeOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_constant_like_conf());
    EnrollInputBn("in", false);
    EnrollOutputBn("out", false);
  }
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().constant_like_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

REGISTER_OP(OperatorConf::kConstantLikeConf, ConstantLikeOp);

}  // namespace oneflow
