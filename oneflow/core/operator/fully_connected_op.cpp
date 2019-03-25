#include "oneflow/core/operator/fully_connected_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void FullyConnectedOp::InitFromOpConf() {
  CHECK(op_conf().has_fully_connected_conf());
  const auto& conf = op_conf().fully_connected_conf();
  EnrollInputBn("in");
  EnrollOutputBn("out");
  if (conf.has_weight()) {
    EnrollModelBn("weight");
  } else {
    EnrollInputBn("weight");
  }

  if (op_conf().fully_connected_conf().use_bias()) {
    if (conf.has_bias()) {
      EnrollModelBn("bias");
    } else {
      EnrollInputBn("bias");
    }
    EnrollConstBufBn("bias_multiplier");
  }
}

const PbMessage& FullyConnectedOp::GetCustomizedConf() const {
  return op_conf().fully_connected_conf();
}

void FullyConnectedOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // useful vars
  const FullyConnectedOpConf& conf = op_conf().fully_connected_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
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
    CHECK_EQ(GetBlobDesc4BnInOp("weight")->shape(), Shape({units, in_blob_desc->shape().Count(1)}));
  } else {
    GetBlobDesc4BnInOp("weight")->mut_shape() = Shape({units, in_blob_desc->shape().Count(1)});
  }

  if (conf.use_bias()) {
    // bias
    if (conf.has_bias()) {
      CHECK_EQ(GetBlobDesc4BnInOp("bias")->shape(), Shape({1, units}));
    } else {
      GetBlobDesc4BnInOp("bias")->mut_shape() = Shape({1, units});
    }

    // bias_multiplier
    GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() = Shape({in_blob_desc->shape().At(0), 1});
  }
}

bool FullyConnectedOp::IsInputBlobAllowedModelSplit(const std::string& ibn) const {
  return ibn == "weight" || ibn == "bias";
}

void FullyConnectedOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  const FullyConnectedOpConf& conf = op_conf().fully_connected_conf();
  if (conf.has_weight()) {
    if (conf.use_bias()) { CHECK(conf.has_bias()); }
    op_parallel_signatures->emplace_back(Make_DS_MB_2_DS_OpParallelSignature(this));
    auto EqZero = [](int32_t x) { return x == 0; };
    op_parallel_signatures->emplace_back(Make_DB_MS_2_MS_OpParallelSignature(this, EqZero));
  } else {
    if (conf.use_bias()) { CHECK(!conf.has_bias()); }
    op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
    op_parallel_signatures->emplace_back(MakeBroadcastOpParallelSignature(this));
  }
}

REGISTER_OP(OperatorConf::kFullyConnectedConf, FullyConnectedOp);

}  // namespace oneflow
