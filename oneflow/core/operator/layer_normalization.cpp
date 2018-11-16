#include "oneflow/core/operator/layer_normalization.h"

namespace oneflow {

void LayerNormOp::InitFromOpConf() {
  CHECK(op_conf().has_layer_norm_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("beta");
  EnrollModelBn("gamma");
}

void LayerNormOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  CHECK(parallel_ctx->policy() != kModelParallel);
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  *out_blob = *in_blob;

  BlobDesc* beta_blob = GetBlobDesc4BnInOp("beta");
  beta_blob->mut_shape() = Shape({in_blob->shape().dim_vec().back()});
  beta_blob->set_data_type(in_blob->data_type());
  *GetBlobDesc4BnInOp("gamma") = *beta_blob;
}

REGISTER_OP(OperatorConf::kLayerNormConf, LayerNormOp);

}  // namespace oneflow
