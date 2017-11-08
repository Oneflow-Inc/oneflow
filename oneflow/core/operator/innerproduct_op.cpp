#include "oneflow/core/operator/innerproduct_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void InnerProductOp::InitFromOpConf() {
  CHECK(op_conf().has_innerproduct_conf());

  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollModelBn("weight");

  if (GetBoolFromSpecialConf("has_bias_term")) {
    EnrollModelBn("bias");
    EnrollModelTmpBn("bias_multiplier");
  }
}

const PbMessage& InnerProductOp::GetSpecialConf() const {
  return op_conf().innerproduct_conf();
}

void InnerProductOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  // useful vars
  const InnerProductOpConf& conf = op_conf().innerproduct_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK_EQ(in_blob_desc->data_type(),
           JobDesc::Singleton()->default_data_type());
  int32_t out_num = conf.out_num();
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(out_num, parallel_ctx->parallel_num());
    out_num = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), out_num});
  out_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  out_blob_desc->set_has_data_id(in_blob_desc->has_data_id());

  // weight
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() =
      Shape({out_num, in_blob_desc->shape().Count(1)});
  weight_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
  weight_blob_desc->set_has_data_id(false);

  if (conf.has_bias_term()) {
    // bias
    BlobDesc* bias_blob_desc = GetBlobDesc4BnInOp("bias");
    bias_blob_desc->mut_shape() = Shape({1, out_num});
    bias_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
    bias_blob_desc->set_has_data_id(false);

    // bias_multiplier
    BlobDesc* bias_mt_blob_desc = GetBlobDesc4BnInOp("bias_multiplier");
    bias_mt_blob_desc->mut_shape() = Shape({in_blob_desc->shape().At(0), 1});
    bias_mt_blob_desc->set_data_type(JobDesc::Singleton()->default_data_type());
    bias_mt_blob_desc->set_has_data_id(false);
  }
}

REGISTER_OP(OperatorConf::kInnerproductConf, InnerProductOp);

}  // namespace oneflow
