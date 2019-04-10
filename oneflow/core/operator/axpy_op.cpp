#include "oneflow/core/operator/axpy_op.h"

namespace oneflow {

namespace {

class AxpySbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AxpySbpSignatureRule);
  ~AxpySbpSignatureRule() override = default;

  AxpySbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (A, A) -> ()"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    if (parallel_desc.parallel_num() != SbpInferHint4Ibn("y").parallel_num()) {
      return MakeSbpSigMatchParallelNumError(parallel_desc.parallel_num(),
                                             SbpInferHint4Ibn("y").parallel_num());
    }
    if (parallel_desc != SbpInferHint4Ibn("y").parallel_desc()) {
      return MakeSbpSigMatchDeviceSetError(parallel_desc.device_names(),
                                           SbpInferHint4Ibn("y").parallel_desc().device_names());
    }
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["y"] = SbpInferHint4Ibn("y").sbp_parallel();
    (*bn2sbp)["x"] = SbpInferHint4Ibn("y").sbp_parallel();
  }
};

}  // namespace

void AxpyOp::InitFromOpConf() {
  EnrollInputBn("y")->set_is_mutable(true);
  EnrollInputBn("x", false);
}

void AxpyOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  CHECK(*GetBlobDesc4BnInOp("x") == *GetBlobDesc4BnInOp("y"));
}

const PbMessage& AxpyOp::GetCustomizedConf() const { return op_conf().axpy_conf(); }

void AxpyOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new AxpySbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kAxpyConf, AxpyOp);

}  // namespace oneflow
