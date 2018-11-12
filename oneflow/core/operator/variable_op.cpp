#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

void VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  EnrollInputBn("tick", false);
  EnrollOutputBn("out", Global<JobDesc>::Get()->IsTrain() && op_conf().trainable());
  EnrollModelBn("weight");
}

const PbMessage& VariableOp::GetCustomizedConf() const { return op_conf().variable_conf(); }

void VariableOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  BlobDesc* weight_blob_desc = GetBlobDesc4BnInOp("weight");
  weight_blob_desc->mut_shape() = Shape(variable_conf.shape());
  weight_blob_desc->set_data_type(variable_conf.has_data_type()
                                      ? variable_conf.data_type()
                                      : Global<JobDesc>::Get()->DefaultDataType());
  *GetBlobDesc4BnInOp("out") = *weight_blob_desc;
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);

}  // namespace oneflow
