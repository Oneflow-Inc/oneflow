#include "oneflow/core/operator/max_pooling_1d_op.h"

namespace oneflow {

const PbMessage& MaxPooling1DOp::GetSpecialConf() const {
  return op_conf().max_pooling_1d_conf();
}

void MaxPooling1DOp::VirtualEnrollDataTmpBn() { EnrollDataTmpBn("idx"); }

void MaxPooling1DOp::VirtualInferDataTmpBlobDesc(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* idx_blob_desc = GetBlobDesc4BnInOp("idx");
  idx_blob_desc->mut_shape() = out_blob_desc->shape();
  idx_blob_desc->set_data_type(DataType::kUInt32);
}

Pooling1DKernelConf* MaxPooling1DOp::GetMutPooling1DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_max_pooling_1d_conf()->mutable_pooling_1d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling1DConf, MaxPooling1DOp);

}  //  namespace oneflow
