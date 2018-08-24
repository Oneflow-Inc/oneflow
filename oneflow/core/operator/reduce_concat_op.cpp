#include "oneflow/core/operator/reduce_concat_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ReduceConcatOp::InitFromOpConf() { TODO(); }

const PbMessage& ReduceConcatOp::GetCustomizedConf() const {
  return op_conf().reduce_scatter_conf();
}

LogicalBlobId ReduceConcatOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

void ReduceConcatOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  TODO();
}

REGISTER_OP(OperatorConf::kReduceConcatConf, ReduceConcatOp);

}  // namespace oneflow
