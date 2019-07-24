#include "oneflow/core/operator/model_init_op.h"

namespace oneflow {

void ModelInitOp::InitFromOpConf() {
  CHECK(op_conf().has_model_init_conf());
  EnrollInputBn("out", false);
}

const PbMessage& ModelInitOp::GetCustomizedConf() const { return op_conf().model_init_conf(); }

void ModelInitOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  const VariableOpConf& variable_conf = op_conf().model_init_conf().variable_conf();
  BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("out");
  model_blob_desc->mut_shape() = Shape(variable_conf.shape());
  model_blob_desc->set_data_type(variable_conf.data_type());
}

REGISTER_OP(OperatorConf::kModelInitConf, ModelInitOp);

}  // namespace oneflow
