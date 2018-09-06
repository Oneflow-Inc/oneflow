#include "oneflow/core/operator/reduce_global_add2_op.h"

namespace oneflow {

void ReduceGlobalAdd2Op::InitFromOpConf() {
  /*
  CHECK(op_conf().has_reduce_global_add2_conf());
  for (int64_t parallel_id : op_conf().reduce_global_add2_conf().in_parallel_ids()) {
    EnrollInputBn("in_" + std::to_string(parallel_id), false);
  }
  EnrollOutputBn("out", false);
  */
}

const PbMessage& ReduceGlobalAdd2Op::GetCustomizedConf() const {
  return op_conf().reduce_global_add2_conf();
}

LogicalBlobId ReduceGlobalAdd2Op::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

void ReduceGlobalAdd2Op::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  /*
  int32_t in_num = op_conf().reduce_global_add2_conf().in_parallel_ids_size();
  CHECK_GE(in_num, 2);
  BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  *GetBlobDesc4BnInOp(SoleObn()) = *first_in_blob;
  for (int32_t i = 1; i < in_num; ++i) {
    CHECK(*first_in_blob == *GetBlobDesc4BnInOp(input_bns().Get(i)));
  }
  */
}

REGISTER_OP(OperatorConf::kReduceGlobalAdd2Conf, ReduceGlobalAdd2Op);

}  // namespace oneflow
