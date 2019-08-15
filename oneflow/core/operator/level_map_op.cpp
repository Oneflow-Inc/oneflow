#include "oneflow/core/operator/level_map_op.h"

namespace oneflow {

void LevelMapOp::InitFromOpConf() {
  CHECK(op_conf().has_level_map_conf());
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& LevelMapOp::GetCustomizedConf() const { return this->op_conf().level_map_conf(); }

void LevelMapOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  // input: boxes
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in->shape().dim_vec().back(), 4);
  // output: levels
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape() =
      Shape(std::vector<int64_t>(in->shape().dim_vec().begin(), in->shape().dim_vec().end() - 1));
  out->set_data_type(DataType::kInt32);
}

void LevelMapOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* kernel_conf) const {
  kernel_conf->set_data_type(GetBlobDesc4BnInOp("in")->data_type());
}

REGISTER_OP(OperatorConf::kLevelMapConf, LevelMapOp);

}  // namespace oneflow
