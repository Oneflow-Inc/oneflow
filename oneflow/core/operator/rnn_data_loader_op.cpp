#include "oneflow/core/operator/rnn_data_loader_op.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void RnnDataLoaderOp::InitFromOpConf() {
  CHECK(op_conf().has_rnn_data_loader_conf());

  EnrollOutputBn("out", false);
}

const PbMessage& RnnDataLoaderOp::GetSpecialConf() const {
  return op_conf().rnn_data_loader_conf();
}

void RnnDataLoaderOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  TODO();
}

REGISTER_OP(OperatorConf::kRnnDataLoaderConf, RnnDataLoaderOp);

}  // namespace oneflow
