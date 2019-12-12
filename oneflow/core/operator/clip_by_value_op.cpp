#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ClipByValueOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipByValueOp);
  ClipByValueOp() = default;
  ~ClipByValueOp() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_clip_by_value_conf());
    EnrollInputBn("in");
    EnrollOutputBn("out");
  }
  const PbMessage& GetCustomizedConf() const override {
    return this->op_conf().clip_by_value_conf();
  }
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    // input
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    // output
    *GetBlobDesc4BnInOp("out") = *in;
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
    return Maybe<void>::Ok();
  }

  Maybe<void> GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override {
    SbpSignatureBuilder().Split("in", 0).Split("out", 0).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kClipByValueConf, ClipByValueOp);

}  // namespace oneflow
