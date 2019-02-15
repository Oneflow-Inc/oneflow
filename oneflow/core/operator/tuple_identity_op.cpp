#include "oneflow/core/operator/tuple_identity_op.h"

namespace oneflow {

void TupleIdentityOp::InitFromOpConf() {
  CHECK(op_conf().has_tuple_identity_conf());
  int32_t in_size = op_conf().tuple_identity_conf().in_size();
  int32_t out_size = op_conf().tuple_identity_conf().out_size();
  CHECK_GT(in_size, 0);
  CHECK_EQ(in_size, out_size);
  EnrollRepeatedInputBn("in", in_size);
  EnrollRepeatedOutputBn("out", out_size);
}

const PbMessage& TupleIdentityOp::GetCustomizedConf() const {
  return op_conf().tuple_identity_conf();
}

void TupleIdentityOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  size_t bn_size = op_conf().tuple_identity_conf().in_size();
  FOR_RANGE(int, i, 0, bn_size) {
    *GetBlobDesc4BnInOp(output_bns().Get(i)) = *GetBlobDesc4BnInOp(input_bns().Get(i));
  }
}

REGISTER_OP(OperatorConf::kTupleIdentityConf, TupleIdentityOp);

}  // namespace oneflow
