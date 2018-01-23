#include "oneflow/core/operator/fully_connected_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void FullyConnectedOp::InitFromOpConf() {
  CHECK(op_conf().has_fully_connected_conf());

  EnrollInputBn("inputs");
  EnrollOutputBn("outputs");
  EnrollModelBn("weights");

  EnrollModelBn("biases");
  EnrollModelTmpBn("biases_multiplier");
}

const PbMessage& FullyConnectedOp::GetSpecialConf() const {
  return op_conf().fully_connected_conf();
}

void FullyConnectedOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // useful vars
  const FullyConnectedOpConf& conf = op_conf().fully_connected_conf();
  const BlobDesc* inputs_blob_desc = GetBlobDesc4BnInOp("inputs");
  CHECK_EQ(inputs_blob_desc->data_type(),
           JobDesc::Singleton()->DefaultDataType());
  int32_t num_outputs = conf.num_outputs();
  if (parallel_ctx->policy() == kModelParallel) {
    BalancedSplitter splitter(num_outputs, parallel_ctx->parallel_num());
    num_outputs = splitter.At(parallel_ctx->parallel_id()).size();
  }
  // outputs
  BlobDesc* outputs_blob_desc = GetBlobDesc4BnInOp("outputs");
  outputs_blob_desc->mut_shape() =
      Shape({inputs_blob_desc->shape().At(0), num_outputs});
  outputs_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  outputs_blob_desc->set_has_data_id_field(
      inputs_blob_desc->has_data_id_field());

  // weights
  BlobDesc* weights_blob_desc = GetBlobDesc4BnInOp("weights");
  weights_blob_desc->mut_shape() =
      Shape({num_outputs, inputs_blob_desc->shape().Count(1)});
  weights_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  weights_blob_desc->set_has_data_id_field(false);

  // biases
  BlobDesc* biases_blob_desc = GetBlobDesc4BnInOp("biases");
  biases_blob_desc->mut_shape() = Shape({1, num_outputs});
  biases_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  biases_blob_desc->set_has_data_id_field(false);

  // biases_multiplier
  BlobDesc* biases_mt_blob_desc = GetBlobDesc4BnInOp("biases_multiplier");
  biases_mt_blob_desc->mut_shape() =
      Shape({inputs_blob_desc->shape().At(0), 1});
  biases_mt_blob_desc->set_data_type(JobDesc::Singleton()->DefaultDataType());
  biases_mt_blob_desc->set_has_data_id_field(false);
}

REGISTER_OP(OperatorConf::kFullyConnectedConf, FullyConnectedOp);

}  // namespace oneflow
