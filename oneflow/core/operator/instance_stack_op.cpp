#include "oneflow/core/operator/instance_stack_op.h"

namespace oneflow {

void InstanceStackOp::InitFromOpConf() {
  CHECK(op_conf().has_instance_stack_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void InstanceStackOp::InferOutputBlobTimeShape(
    std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
    const ParallelContext* parallel_ctx, Shape* time_shape) const {
  std::vector<int64_t> dim_vec(GetTimeShape4BnInOp("in")->dim_vec());
  dim_vec.pop_back();
  *time_shape = Shape(dim_vec);
}

REGISTER_OP(OperatorConf::kInstanceStackConf, InstanceStackOp);

}  // namespace oneflow
