#include "oneflow/core/operator/model_load_op.h"

namespace oneflow {

void ModelLoadOp::InitFromOpConf() {
  CHECK(op_conf().has_model_load_conf());
  EnrollInputBn("out", false);
}

const PbMessage& ModelLoadOp::GetCustomizedConf() const { return op_conf().model_load_conf(); }

void ModelLoadOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  const ModelLoadOpConf &conf = op_conf().model_load_conf();
  BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("out");
  model_blob_desc->mut_shape() = Shape(conf.shape());
  DataType data_type = Global<JobDesc>::Get()->DefaultDataType();
  if (conf.has_data_type()) {
    data_type = conf.data_type();
  }
  model_blob_desc->set_data_type(data_type);
}

REGISTER_OP(OperatorConf::kModelLoadConf, ModelLoadOp);

}  // namespace oneflow
