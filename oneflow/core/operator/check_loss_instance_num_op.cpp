#include "oneflow/core/operator/check_loss_instance_num_op.h"

namespace oneflow {

namespace {

class CheckLossInstanceOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CheckLossInstanceOpParallelSignature);
  ~CheckLossInstanceOpParallelSignature() override = default;

  CheckLossInstanceOpParallelSignature(const Operator* op) : OpParallelSignature(op) {}

  const std::string Description() const override { return op().op_name() + ": (B, ...) -> (B,)"; }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    for (const std::string& ibn : op().input_bns()) {
      if (!SbpInferHint4Ibn(ibn).sbp_parallel().has_partial_sum_parallel()) {
        return MakeOpParallelMatchSignatureMismatch();
      }
    }
    return MakeOpParallelMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : op().input_bns()) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
    for (const auto& bn : op().output_bns()) { (*bn2sbp)[bn].mutable_broadcast_parallel(); }
  }
};

}  // namespace

void CheckLossInstanceNumOp::VirtualInitFromOpConf() {
  CHECK(op_conf().has_check_loss_instance_num_conf());
}

void CheckLossInstanceNumOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  for (const std::string& ibn : input_bns()) {
    CHECK(*GetBlobDesc4BnInOp(ibn) == *GetBlobDesc4BnInOp(input_bns().Get(0)));
  }
}

const PbMessage& CheckLossInstanceNumOp::GetCustomizedConf() const {
  return op_conf().check_loss_instance_num_conf();
}

void CheckLossInstanceNumOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(new CheckLossInstanceOpParallelSignature(this));
}

REGISTER_CPU_OP(OperatorConf::kCheckLossInstanceNumConf, CheckLossInstanceNumOp);

}  // namespace oneflow
