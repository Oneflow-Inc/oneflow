#include "oneflow/core/operator/calculate_scale_op.h"

namespace oneflow {

void CalculateScaleOp::InitFromOpConf() {
  CHECK(op_conf().has_calculate_scale_conf());
  EnrollInputBn("origin_height", false);
  EnrollInputBn("origin_width", false);
  EnrollOutputBn("scale", false);
  EnrollOutputBn("height", false);
  EnrollOutputBn("width", false);
}

void CalculateScaleOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().calculate_scale_conf();
  const int32_t target_size = conf.target_size();
  const int32_t max_size = conf.max_size();
  CHECK_GT(target_size, 0);
  CHECK_GE(max_size, target_size);
  const BlobDesc* origin_height = GetBlobDesc4BnInOp("origin_height");
  const BlobDesc* origin_width = GetBlobDesc4BnInOp("origin_width");
  CHECK_EQ(origin_height->shape().NumAxes(), 1);
  CHECK_EQ(origin_height->shape(), origin_width->shape());
  CHECK_EQ(origin_height->data_type(), DataType::kInt32);
  CHECK_EQ(origin_height->data_type(), origin_width->data_type());

  // scale
  BlobDesc* scale = GetBlobDesc4BnInOp("scale");
  scale->mut_shape() = origin_height->shape();
  scale->set_data_type(Global<JobDesc>::Get()->DefaultDataType());

  // height & width
  *GetBlobDesc4BnInOp("height") = *origin_height;
  *GetBlobDesc4BnInOp("width") = *origin_width;
}

void CalculateScaleOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("scale")->data_type());
}

REGISTER_CPU_OP(OperatorConf::kCalculateScaleConf, CalculateScaleOp);

}  // namespace oneflow
