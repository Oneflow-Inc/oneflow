#include "oneflow/core/operator/reduce_local_add_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ReduceLocalAddOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_local_add_conf());
  FOR_RANGE(int32_t, i, 0, op_conf().reduce_local_add_conf().in_num()) {
    EnrollInputBn("in_" + std::to_string(i), false);
  }
  EnrollDataTmpBn("middle");
  FOR_RANGE(int32_t, i, 0, op_conf().reduce_local_add_conf().out_num()) {
    EnrollOutputBn("out_" + std::to_string(i), false);
  }
}

const PbMessage& ReduceLocalAddOp::GetCustomizedConf() const {
  return op_conf().reduce_local_add_conf();
}

LogicalBlobId ReduceLocalAddOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

void ReduceLocalAddOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  int32_t in_num = op_conf().reduce_local_add_conf().in_num();
  // TODO check in_num = device used in this computer
  CHECK_GE(in_num, 2);
  const BlobDesc* first_in_blob = GetBlobDesc4BnInOp(input_bns().Get(0));
  FOR_RANGE(int32_t, i, 1, in_num) {
    CHECK(*first_in_blob == *GetBlobDesc4BnInOp(input_bns().Get(i)));
  }
  *GetBlobDesc4BnInOp("middle") = *first_in_blob;
  BalancedSplitter splitter(op_conf().reduce_local_add_conf().model_elem_cnt(),
                            parallel_ctx->parallel_num());
  int32_t first_parallel_id = op_conf().reduce_local_add_conf().first_parallel_id();
  FOR_RANGE(int32_t, i, 0, op_conf().reduce_local_add_conf().out_num()) {
    BlobDesc* out_blob_i = GetBlobDesc4BnInOp("out_" + std::to_string(i));
    *out_blob_i = *first_in_blob;
    out_blob_i->mut_shape() = Shape({splitter.At(first_parallel_id + i).size()});
  }
}

REGISTER_OP(OperatorConf::kReduceLocalAddConf, ReduceLocalAddOp);

}  // namespace oneflow
