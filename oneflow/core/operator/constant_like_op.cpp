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
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    const auto& bn2conf_sbp = sbp_sig_conf.bn_in_op2sbp_parallel();
    const SbpParallel* sbp_parallel = nullptr;
    const auto& conf_sbp_it = bn2conf_sbp.find("out");
    if (conf_sbp_it == bn2conf_sbp.end()) {
      sbp_parallel = &(JUST(SbpInferHint4Ibn("in"))->sbp_parallel());
    } else {
      sbp_parallel = &conf_sbp_it->second;
    }
    (*bn2sbp)["in"] = *sbp_parallel;
    (*bn2sbp)["out"] = *sbp_parallel;
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kConstantLikeConf, ConstantLikeOp);

}  // namespace oneflow
