#include "oneflow/core/operator/fully_connected_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void FullyConnectedOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("weight");

  if (GetValFromCustomizedConf<bool>("use_bias")) {
    EnrollModelBn("bias");
    EnrollModelTmpBn("bias_multiplier");
  }
  if (GetValFromCustomizedConf<bool>("has_activation")) {
    EnrollDataTmpBn("activation_buf");
  }
}

const PbMessage& FullyConnectedOp::GetCustomizedConf() const {
  return op_conf().fully_connected_conf();
}

void FullyConnectedOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // useful vars
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->data_type(),
           Global<JobDesc>::Get()->DefaultDataType());
  int32_t units = GetValFromCustomizedConf<int32_t>("units");
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(units, parallel_ctx->parallel_num());
    units = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), units});
  if (GetValFromCustomizedConf<bool>("has_activation")) {
    GetBlobDesc4BnInOp("activation_buf")->mut_shape() = out_blob_desc->shape();
  }

  // weight
  GetBlobDesc4BnInOp("weight")->mut_shape() =
      Shape({units, in_blob_desc->shape().Count(1)});

  if (GetValFromCustomizedConf<bool>("use_bias")) {
    // bias
    GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({1, units});

    // bias_multiplier
    GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() =
        Shape({in_blob_desc->shape().At(0), 1});
  }
}

REGISTER_OP(OperatorConf::kFullyConnectedConf, FullyConnectedOp);

}  // namespace oneflow
