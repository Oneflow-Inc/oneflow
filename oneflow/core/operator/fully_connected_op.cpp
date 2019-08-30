#include "oneflow/core/operator/fully_connected_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void FullyConnectedOp::InitFromOpConf() {
  CHECK(op_conf().has_fully_connected_conf());
  const auto& conf = op_conf().fully_connected_conf();
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (conf.has_weight()) {
    EnrollInputBn("weight");
  } else {
    EnrollTmpBn("weight");
  }

  if (op_conf().fully_connected_conf().use_bias()) {
    if (conf.has_bias()) {
      EnrollInputBn("bias");
    } else {
      EnrollTmpBn("bias");
    }
    EnrollConstBufBn("bias_multiplier");
  }
}

const PbMessage& FullyConnectedOp::GetCustomizedConf() const {
  return op_conf().fully_connected_conf();
}

Maybe<void> FullyConnectedOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // useful vars
  const FullyConnectedOpConf& conf = op_conf().fully_connected_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ_OR_RETURN(in_blob_desc->data_type(), GlobalJobDesc().DefaultDataType());
  int32_t units = conf.units();
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(units, parallel_ctx->parallel_num());
    units = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), units});

  // weight
  if (conf.has_weight()) {
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("weight")->shape(),
                       Shape({units, in_blob_desc->shape().Count(1)}));
  } else {
    GetBlobDesc4BnInOp("weight")->mut_shape() = Shape({units, in_blob_desc->shape().Count(1)});
  }

  if (conf.use_bias()) {
    // bias
    if (conf.has_bias()) {
      CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("bias")->shape(), Shape({1, units}));
    } else {
      GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({1, units});
    }

    // bias_multiplier
    GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() = Shape({in_blob_desc->shape().At(0), 1});
  }
  return Maybe<void>::Ok();
}

void FullyConnectedOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int32_t num_axes = LogicalBlobDesc4Ibn("in").shape().NumAxes();
  SbpSignatureBuilder()
      .Split("in", 0)
      .Broadcast({"weight", "bias"})
      .Split("out", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());

  SbpSignatureBuilder()
      .Broadcast("in")
      .Split({"weight", "bias"}, 0)
      .Split("out", num_axes - 1)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kFullyConnectedConf, FullyConnectedOp);

}  // namespace oneflow
