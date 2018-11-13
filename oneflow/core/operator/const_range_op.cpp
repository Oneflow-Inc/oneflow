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
  int64_t size = 0;
  if (conf.has_const_size()) {
    size = conf.const_size();
  } else if (conf.has_use_device_piece_size()) {
    CHECK_EQ(Global<JobDesc>::Get()->PieceSize() % parallel_ctx->parallel_num(), 0);
    size = Global<JobDesc>::Get()->PieceSize() / parallel_ctx->parallel_num();
  } else {
    UNIMPLEMENTED();
  }
  CHECK_GE(size, 1);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(conf.data_type());
  out->mut_shape() = Shape({size});
}

REGISTER_OP(OperatorConf::kConstRangeConf, ConstRangeOp);

}  // namespace oneflow
