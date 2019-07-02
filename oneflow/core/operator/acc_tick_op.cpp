#include "oneflow/core/operator/acc_tick_op.h"

namespace oneflow {

void AccTickOp::InitFromOpConf() {
  CHECK(op_conf().has_acc_tick_conf());

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

void AccTickOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("acc") = *GetBlobDesc4BnInOp("one");
  GetBlobDesc4BnInOp("acc")->mut_shape() = Shape({1LL});
}

void AccTickOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  const int32_t max_acc_num = op_conf().acc_tick_conf().max_acc_num();
  CHECK_EQ(GetTimeShape4BnInOp("one")->elem_cnt() % max_acc_num, 0);
  *time_shape = Shape({GetTimeShape4BnInOp("one")->elem_cnt() / max_acc_num});
}

const PbMessage& AccTickOp::GetCustomizedConf() const { return op_conf().acc_tick_conf(); }

void AccTickOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("acc") = false;
}

REGISTER_OP(OperatorConf::kAccTickConf, AccTickOp);

}  // namespace oneflow
