#include "oneflow/core/operator/max_pooling_2d_op.h"

namespace oneflow {

const PbMessage& MaxPooling2DOp::GetSpecialConf() const {
  return op_conf().max_pooling_2d_conf();
}

void MaxPooling2DOp::VirtualEnrollDataTmpBn() { EnrollDataTmpBn("idx"); }

void MaxPooling2DOp::VirtualInferDataTmpBlobDesc(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* idx_blob_desc = GetBlobDesc4BnInOp("idx");
  idx_blob_desc->mut_shape() = out_blob_desc->shape();
  idx_blob_desc->set_data_type(DataType::kInt32);
}

Pooling2DKernelConf* MaxPooling2DOp::GetMutPooling2DKernelConf(
    KernelConf* kernel_conf) const {
  return kernel_conf->mutable_max_pooling_2d_conf()->mutable_pooling_2d_conf();
}

REGISTER_OP(OperatorConf::kMaxPooling2DConf, MaxPooling2DOp);

}  //  namespace oneflow
