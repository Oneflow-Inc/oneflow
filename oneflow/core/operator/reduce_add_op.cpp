#include "oneflow/core/operator/reduce_add_op.h"

namespace oneflow {

void ReduceAddOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_add_conf());
  FOR_RANGE(int32_t, i, 0, op_conf().reduce_add_conf().in_num()) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollOutputBn("out", false);
}

const PbMessage& ReduceAddOp::GetCustomizedConf() const { return op_conf().reduce_add_conf(); }

LogicalBlobId ReduceAddOp::lbi4obn(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

Maybe<void> ReduceAddOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t in_num = op_conf().reduce_add_conf().in_num();
  CHECK_GE_OR_RETURN(in_num, 2);
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  *GetBlobDesc4BnInOp(SoleObn()) = *first_in_blob;
  for (int32_t i = 1; i < in_num; ++i) {
    CHECK_OR_RETURN(*first_in_blob == *GetBlobDesc4BnInOp(input_bns().Get(i)));
  }
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> ReduceAddOp::GetOpConfWithoutOpNameAndLbn() const {
  OperatorConf op_conf(this->op_conf());
  op_conf.set_name("");
  return SymbolOf(op_conf);
}

REGISTER_OP(OperatorConf::kReduceAddConf, ReduceAddOp);

}  // namespace oneflow
