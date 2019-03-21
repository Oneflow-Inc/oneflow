#include "oneflow/core/operator/loss_print_op.h"

namespace oneflow {

namespace {

class LossPrintOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LossPrintOpParallelSignature);
  ~LossPrintOpParallelSignature() override = default;

  LossPrintOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (B, B) -> B"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4BnInOp,
      const ParallelDesc& parallel_desc) const override {
    return MakeOpParallelMatchSuccess();
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

void LossPrintOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new LossPrintOpParallelSignature(this));
}

REGISTER_CPU_OP(OperatorConf::kLossPrintConf, LossPrintOp);

}  // namespace oneflow
