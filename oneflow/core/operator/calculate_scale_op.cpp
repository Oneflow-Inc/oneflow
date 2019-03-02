#include "oneflow/core/operator/calculate_scale_op.h"

namespace oneflow {

void CalculateScaleOp::InitFromOpConf() {
  CHECK(op_conf().has_calculate_scale_conf());
  EnrollInputBn("height", false);
  EnrollInputBn("weight", false);
  EnrollOutputBn("scale", false);
}

void CalculateScaleOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().calculate_scale_conf();
  const int32_t target_size = conf.target_size();
  const int32_t max_size = conf.max_size();
  CHECK_GT(target_size, 0);
  CHECK_GE(max_size, target_size);
  const BlobDesc* height = GetBlobDesc4BnInOp("height");
  const BlobDesc* weight = GetBlobDesc4BnInOp("weight");
  CHECK_EQ(height->shape().NumAxes(), 1);
  CHECK_EQ(height->shape(), weight->shape());
  CHECK_EQ(height->data_type(), DataType::kInt32);
  CHECK_EQ(weight->data_type(), weight->data_type());

  // scale
  BlobDesc* scale = GetBlobDesc4BnInOp("scale");
  scale->mut_shape() = height->shape();
  scale->set_data_type(Global<JobDesc>::Get()->DefaultDataType());
}

REGISTER_CPU_OP(OperatorConf::kCalculateScaleConf, CalculateScaleOp);

}  // namespace oneflow
