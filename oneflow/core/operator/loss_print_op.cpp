#include "oneflow/core/operator/loss_print_op.h"

namespace oneflow {

namespace {

class LossPrintSbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossPrintSbpSignatureRule);
  ~LossPrintSbpSignatureRule() override = default;

  LossPrintSbpSignatureRule(const Operator* op) : ParallelSbpSignatureRule(op) {}

  const std::string Description() const override { return op().op_name() + ": (B, B) -> B"; }

  const SbpSigMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    (*bn2sbp)["loss_acc"].mutable_broadcast_parallel();
    (*bn2sbp)["loss_instance_num"].mutable_broadcast_parallel();
    if (op().op_conf().loss_print_conf().has_reduction_lbi()) {
      (*bn2sbp)["reduction_acc"].mutable_broadcast_parallel();
    }
  }
};

}  // namespace

void LossPrintOp::InitFromOpConf() {
  CHECK(op_conf().has_loss_print_conf());
  EnrollInputBn("loss_acc", false);
  EnrollInputBn("loss_instance_num", false);
  if (op_conf().loss_print_conf().has_reduction_lbi()) { EnrollInputBn("reduction_acc"); }
}

void LossPrintOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  CHECK_EQ(GetBlobDesc4BnInOp("loss_acc")->shape().NumAxes(), 1);
  CHECK_EQ(GetBlobDesc4BnInOp("loss_instance_num")->shape().NumAxes(), 1);
  if (op_conf().loss_print_conf().has_reduction_lbi()) {
    CHECK_EQ(GetBlobDesc4BnInOp("reduction_acc")->shape().NumAxes(), 1);
  }
}

LogicalBlobId LossPrintOp::ibn2lbi(const std::string& input_bn) const {
  if (input_bn == "loss_acc") {
    return op_conf().loss_print_conf().loss_lbi();
  } else if (input_bn == "loss_instance_num") {
    return op_conf().loss_print_conf().loss_instance_num_lbi();
  } else if (input_bn == "reduction_acc") {
    return op_conf().loss_print_conf().reduction_lbi();
  } else {
    UNIMPLEMENTED();
  }
}

const PbMessage& LossPrintOp::GetCustomizedConf() const { return op_conf().loss_print_conf(); }

void LossPrintOp::GetSbpSignatureRules(
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  rules->emplace_back(new LossPrintSbpSignatureRule(this));
}

REGISTER_CPU_OP(OperatorConf::kLossPrintConf, LossPrintOp);

}  // namespace oneflow
