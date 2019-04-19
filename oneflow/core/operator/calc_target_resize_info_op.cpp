#include "oneflow/core/operator/calc_target_resize_info_op.h"

namespace oneflow {

void CalcTargetResizeInfoOp::InitFromOpConf() {
  CHECK(op_conf().has_calc_target_resize_info_conf());
  EnrollInputBn("origin_height", false);
  EnrollInputBn("origin_width", false);
  EnrollDataTmpBn("scale");
  EnrollOutputBn("resized_image_size", false);
}

void CalcTargetResizeInfoOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().calc_target_resize_info_conf();
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

  // resized_image_size
  BlobDesc* resized_image_size = GetBlobDesc4BnInOp("resized_image_size");
  resized_image_size->mut_shape() = Shape({origin_height->shape().At(0), 2});
  resized_image_size->set_data_type(origin_height->data_type());
}

REGISTER_CPU_OP(OperatorConf::kCalcTargetResizeInfoConf, CalcTargetResizeInfoOp);

}  // namespace oneflow
