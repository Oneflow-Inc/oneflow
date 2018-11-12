#include "oneflow/core/operator/const_range_op.h"

namespace oneflow {

void ConstRangeOp::InitFromOpConf() {
  CHECK(op_conf().has_const_range_conf());
  EnrollInputBn("tick", false);
  EnrollOutputBn("out", false);
}

const PbMessage& ConstRangeOp::GetCustomizedConf() const { return op_conf().const_range_conf(); }

void ConstRangeOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  CHECK_EQ(parallel_ctx->policy(), ParallelPolicy::kDataParallel);
  const ConstRangeOpConf& conf = op_conf().const_range_conf();
  CHECK(IsIntegralDataType(conf.data_type()));
  CHECK_GE(conf.size(), 1);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(conf.data_type());
  out->mut_shape() = Shape({conf.size()});
}

REGISTER_OP(OperatorConf::kConstRangeConf, ConstRangeOp);

}  // namespace oneflow
