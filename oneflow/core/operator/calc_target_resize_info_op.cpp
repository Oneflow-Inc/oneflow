#include "oneflow/core/operator/calc_target_resize_info_op.h"

namespace oneflow {

void CalcTargetResizeInfoOp::InitFromOpConf() {
  CHECK(op_conf().has_calc_target_resize_info_conf());
  EnrollInputBn("in", false);
  EnrollDataTmpBn("scale");
  EnrollOutputBn("out", false);
}

void CalcTargetResizeInfoOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& conf = op_conf().calc_target_resize_info_conf();
  const int32_t target_size = conf.target_size();
  const int32_t max_size = conf.max_size();
  CHECK_GT(target_size, 0);
  CHECK_GE(max_size, target_size);

  // in: (N, 2)
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in->shape().NumAxes(), 2);
  CHECK_EQ(in->shape().At(1), 2);
  CHECK_EQ(in->data_type(), DataType::kInt32);

  // data tmp: scale (N,)
  BlobDesc* scale = GetBlobDesc4BnInOp("scale");
  scale->mut_shape() = Shape({in->shape().At(0)});
  scale->set_data_type(Global<JobDesc>::Get()->DefaultDataType());

  // out: (N, 2)
  *GetBlobDesc4BnInOp("out") = *in;
}

REGISTER_CPU_OP(OperatorConf::kCalcTargetResizeInfoConf, CalcTargetResizeInfoOp);

}  // namespace oneflow
