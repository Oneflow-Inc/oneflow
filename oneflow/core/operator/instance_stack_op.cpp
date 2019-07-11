#include "oneflow/core/operator/instance_stack_op.h"

namespace oneflow {

void InstanceStackOp::InitFromOpConf() {
  CHECK(op_conf().has_instance_stack_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

// FIX ME: instance_stack op override this function just in order to satisfy op_graph, remove later
void InstanceStackOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp(SoleIbn());
  BlobDesc* out = GetBlobDesc4BnInOp(SoleObn());
  std::vector<int64_t> dim_vec = in->shape().dim_vec();
  dim_vec.insert(dim_vec.begin(), op_conf().instance_stack_conf().stack_num());
  out->mut_shape() = Shape(dim_vec);
  out->set_data_type(in->data_type());
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
