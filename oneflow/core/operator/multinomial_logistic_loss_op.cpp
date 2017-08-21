#include "oneflow/core/operator/multinomial_logistic_loss_op.h"

namespace oneflow {

void MultinomialLogisticLossOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_multinomial_logistic_loss_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("loss", false);
  EnrollDataTmpBn("loss_buffer");
}

const PbMessage& MultinomialLogisticLossOp::GetSpecialConf() const {
  return op_conf().multinomial_logistic_loss_conf();
}

void MultinomialLogisticLossOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  GetBlobDesc4BnInOp("loss")->mut_shape() = Shape({1});
  GetBlobDesc4BnInOp("loss_buffer")->mut_shape() = Shape({1});
}

REGISTER_OP(OperatorConf::kMultinomialLogisticLossConf,
            MultinomialLogisticLossOp);

}  // namespace oneflow
