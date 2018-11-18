#include "oneflow/core/operator/mean_op.h"

namespace oneflow {

void MeanOp::InitFromOpConf() {
  CHECK(op_conf().has_mean_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollConstBufBn("mean_multiplier");
  EnrollBwBufBn("bw_tmp");
  EnrollFwBufBn("fw_tmp");
}

void MeanOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  *out_blob = *in_blob;
  std::vector<int64_t> dim_vec = in_blob->shape().dim_vec();
  int64_t reduced_dim_size = dim_vec.back();  // only calc mean on final dim now
  dim_vec.back() = 1;
  out_blob->mut_shape() = Shape(std::move(dim_vec));
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
  GetBlobDesc4BnInOp("mean_multiplier")->mut_shape() = Shape({reduced_dim_size, 1});
}

void MeanOp::InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext*) const {
  const BlobDesc* out = GetBlobDesc4BnInOp("out");
  BlobDesc* bw_tmp = GetBlobDesc4BnInOp("bw_tmp");
  bw_tmp->mut_shape() = Shape({out->shape().elem_cnt()});
  bw_tmp->set_data_type(out->data_type());
}

REGISTER_OP(OperatorConf::kMeanConf, MeanOp);

}  // namespace oneflow
