#include "oneflow/core/operator/reduce_split_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ReduceSplitOp::InitFromOpConf() { TODO(); }

const PbMessage& ReduceSplitOp::GetCustomizedConf() const { return op_conf().reduce_split_conf(); }

LogicalBlobId ReduceSplitOp::obn2lbi(const std::string& output_bn) const {
  LogicalBlobId ret;
  ret.set_op_name(op_name());
  ret.set_blob_name(output_bn);
  return ret;
}

void ReduceSplitOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  TODO();
}

REGISTER_OP(OperatorConf::kReduceSplitConf, ReduceSplitOp);

}  // namespace oneflow
