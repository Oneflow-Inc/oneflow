#include "oneflow/core/operator/max_pooling_op.h"

namespace oneflow {

const PbMessage& MaxPoolingOp::GetSpecialConf() const {
  return op_conf().max_pooling_conf();
}

void MaxPoolingOp::VirtualEnrollDataTmpBn() { EnrollDataTmpBn("idx"); }

void MaxPoolingOp::VirtualInferDataTmpBlobDesc(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* idx_blob_desc = GetBlobDesc4BnInOp("idx");
  idx_blob_desc->mut_shape() = out_blob_desc->shape();
  idx_blob_desc->set_data_type(DataType::kUInt32);
}

REGISTER_OP(OperatorConf::kMaxPoolingConf, MaxPoolingOp);

}  //  namespace oneflow
