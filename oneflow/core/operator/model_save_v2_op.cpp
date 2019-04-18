#include "oneflow/core/operator/model_save_v2_op.h"

namespace oneflow {

namespace {

class ModelSaveV2SbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveV2SbpSignatureRule);
  ~ModelSaveV2SbpSignatureRule() override = default;

  explicit ModelSaveV2SbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (B,) -> ()"; }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["in"].mutable_broadcast_parallel();
  }
};

}  // namespace

void ModelSaveV2Op::InitFromOpConf() {
  CHECK(op_conf().has_model_save_v2_conf());
  EnrollInputBn("in", false);
}

const PbMessage& ModelSaveV2Op::GetCustomizedConf() const { return op_conf().model_save_v2_conf(); }

void ModelSaveV2Op::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new ModelSaveV2SbpSignatureRule(this));
}

REGISTER_OP(OperatorConf::kModelSaveV2Conf, ModelSaveV2Op);

}  // namespace oneflow
