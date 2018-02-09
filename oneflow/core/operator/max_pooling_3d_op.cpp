#include "oneflow/core/operator/max_pooling_3d_op.h"

namespace oneflow {

const PbMessage& MaxPooling3DOp::GetSpecialConf() const {
  return op_conf().max_pooling_3d_conf();
}

void MaxPooling3DOp::VirtualEnrollDataTmpBn() { EnrollDataTmpBn("idx"); }

void MaxPooling3DOp::VirtualInferDataTmpBlobDesc(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* idx_blob_desc = GetBlobDesc4BnInOp("idx");
  idx_blob_desc->mut_shape() = out_blob_desc->shape();
  idx_blob_desc->set_data_type(DataType::kUInt32);
}

Pooling3DKernelConf* MaxPooling3DOp::GetMutPooling3DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_max_pooling_3d_conf()->mutable_pooling_3d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling3DConf, MaxPooling3DOp);

}  //  namespace oneflow
