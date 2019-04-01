#include "oneflow/core/operator/acc_op.h"

namespace oneflow {

void AccOp::InitFromOpConf() {
  CHECK(op_conf().has_acc_conf());

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

void AccOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  const int32_t max_acc_num = op_conf().acc_conf().max_acc_num();
  CHECK_GE(GetTimeShape4BnInOp("one")->elem_cnt(), max_acc_num);
  *time_shape = Shape({GetTimeShape4BnInOp("one")->elem_cnt() / max_acc_num});
}

const PbMessage& AccOp::GetCustomizedConf() const { return op_conf().acc_conf(); }

REGISTER_OP(OperatorConf::kAccConf, AccOp);

}  // namespace oneflow
