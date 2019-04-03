#include "oneflow/core/operator/model_save_v2_op.h"

namespace oneflow {

namespace {

class ModelSaveV2SbpSignature final : public ParallelSbpSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveV2SbpSignature);
  ~ModelSaveV2SbpSignature() override = default;

  explicit ModelSaveV2SbpSignature(const Operator* op) : ParallelSbpSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (B,) -> ()"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["in"].mutable_broadcast_parallel();
  }
};

}  // namespace

void ModelSaveV2Op::InitFromOpConf() {
  CHECK(op_conf().has_model_save_v2_conf());
  EnrollInputBn("in", false);
}

const PbMessage& ModelSaveV2Op::GetCustomizedConf() const { return op_conf().model_save_v2_conf(); }

void ModelSaveV2Op::GetSbpSignatures(
    std::vector<std::unique_ptr<const SbpSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new ModelSaveV2SbpSignature(this));
}

REGISTER_OP(OperatorConf::kModelSaveV2Conf, ModelSaveV2Op);

}  // namespace oneflow
