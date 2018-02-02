#include "oneflow/core/operator/reshape_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void ReshapeOp::InitFromOpConf() {
  CHECK(op_conf().has_reshape_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& ReshapeOp::GetSpecialConf() const {
  return op_conf().reshape_conf();
}

void ReshapeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *GetBlobDesc4BnInOp("in");

  const ReshapeOpConf& conf = op_conf().reshape_conf();
  std::vector<int64_t> dim_vec(1 + conf.shape().dim_size());
  dim_vec[0] = JobDesc::Singleton()->SinglePieceSize();
  FOR_RANGE(size_t, i, 1, dim_vec.size()) {
    dim_vec[i] = conf.shape().dim(i - 1);
  }
  out_blob_desc->mut_shape() = Shape(dim_vec);
}

REGISTER_OP(OperatorConf::kReshapeConf, ReshapeOp);

}  // namespace oneflow
