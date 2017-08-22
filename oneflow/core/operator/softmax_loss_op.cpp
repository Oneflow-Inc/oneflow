#include "oneflow/core/operator/softmax_loss_op.h"

namespace oneflow {

void SoftmaxLossOp::InitFromOpConf(const OperatorConf& op_conf) {
  CHECK(op_conf.has_softmax_loss_conf());
  mut_op_conf() = op_conf;

  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollDataTmpBn("prob");
  EnrollDataTmpBn("tmp_1D");
  EnrollOutputBn("loss", false);
}

const PbMessage& SoftmaxLossOp::GetSpecialConf() const {
  return op_conf().softmax_loss_conf();
}

void SoftmaxLossOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) const {
  const std::vector<int64_t> in_dim_vec =
      GetBlobDesc4BnInOp("prediction")->shape().dim_vec();
  CHECK_EQ(in_dim_vec.size(), 2);
  CHECK_EQ(GetBlobDesc4BnInOp("label")->shape(), Shape({in_dim_vec[0]}));
  GetBlobDesc4BnInOp(SoleObn())->mut_shape() = Shape({1});
  GetBlobDesc4BnInOp("prob")->mut_shape() = Shape(in_dim_vec);
  GetBlobDesc4BnInOp("tmp_1D")->mut_shape() = Shape({in_dim_vec[0]});
}

REGISTER_OP(OperatorConf::kSoftmaxLossConf, SoftmaxLossOp);

}  // namespace oneflow
