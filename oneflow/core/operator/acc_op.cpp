#include "oneflow/core/operator/acc_op.h"

namespace oneflow {

void AccOp::InitFromOpConf() {
  CHECK(op_conf().has_acc_conf());

  EnrollInputBn("one", false);
  EnrollOutputBn("acc", false);
}

Maybe<void> AccOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("acc") = *GetBlobDesc4BnInOp("one");
  return Maybe<void>::Ok();
}

Maybe<void> AccOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  const int32_t max_acc_num = op_conf().acc_conf().max_acc_num();
  // CHECK_GE(GetTimeShape4BnInOp("one")->elem_cnt(), max_acc_num);
  CHECK_GE_OR_RETURN(GetTimeShape4BnInOp("one")->elem_cnt(), max_acc_num, "");
  *time_shape = Shape({GetTimeShape4BnInOp("one")->elem_cnt() / max_acc_num});
  return Maybe<void>::Ok();
}

const PbMessage& AccOp::GetCustomizedConf() const { return op_conf().acc_conf(); }

Maybe<void> AccOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("acc") = false;
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kAccConf, AccOp);

}  // namespace oneflow
