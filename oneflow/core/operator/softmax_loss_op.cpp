#include "oneflow/core/operator/softmax_loss_op.h"

namespace oneflow {

void SoftmaxLossOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_softmax_loss_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("in");
  EnrollInputBn("label", false);
  EnrollDataTmpBn("prob");
  EnrollDataTmpBn("tmp_1D");
  EnrollOutputBn("loss", false);
}

const PbMessage& SoftmaxLossOp::GetSpecialConf() const {
  return op_conf().softmax_loss_conf();
}

void SoftmaxLossOp::InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  const std::vector<int64_t> in_dim_vec = GetShapePtr4BnInOp("in")->dim_vec();
  CHECK_EQ(in_dim_vec.size(), 2);
  CHECK_EQ(*GetShapePtr4BnInOp("label"), Shape({in_dim_vec[0]}));
  *GetShapePtr4BnInOp(SoleObn()) = Shape({1});
  *GetShapePtr4BnInOp("prob") = Shape(in_dim_vec);
  *GetShapePtr4BnInOp("tmp_1D") = Shape({in_dim_vec[0]});
}

REGISTER_OP(OperatorConf::kSoftmaxLossConf, SoftmaxLossOp);

}  // namespace oneflow
