#include "oneflow/core/operator/stochastic_pooling_op.h"

namespace oneflow {

const PbMessage& StochasticPoolingOp::GetSpecialConf() const {
  return op_conf().stochastic_pooling_conf();
}

void StochasticPoolingOp::VirtualEnrollDataTmpBn() { EnrollDataTmpBn("idx"); }

void StochasticPoolingOp::VirtualInferDataTmpBlobDesc(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* idx_blob_desc = GetBlobDesc4BnInOp("idx");
  idx_blob_desc->mut_shape() = out_blob_desc->shape();
  idx_blob_desc->set_data_type(DataType::kUInt32);
}

REGISTER_OP(OperatorConf::kMaxPoolingConf, StochasticPoolingOp);

}  //  namespace oneflow
