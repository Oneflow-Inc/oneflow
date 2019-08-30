#include "oneflow/core/operator/hinge_loss_op.h"

namespace oneflow {

void HingeLossOp::VirtualInitFromOpConf() {
  EnrollTmpBn("tmp");
  EnrollTmpBn("tmp_diff");
  EnrollTmpBn("tmp_storage");  // used by GPU
}

const PbMessage& HingeLossOp::GetCustomizedConf() const { return op_conf().hinge_loss_conf(); }

LossKernelConf* HingeLossOp::GetMutLossKernelConf(KernelConf* kernel_conf) const {
  return kernel_conf->mutable_hinge_loss_conf()->mutable_loss_conf();
}

Maybe<void> HingeLossOp::VirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
#define OF_HINGE_LOSS_INFER_TMP_BLOB_DESC(blobname)                   \
  BlobDesc* blobname##_blob_desc = GetBlobDesc4BnInOp(#blobname);     \
  blobname##_blob_desc->mut_shape() = Shape(pred_blob_desc->shape()); \
  blobname##_blob_desc->set_data_type(pred_blob_desc->data_type())
  OF_HINGE_LOSS_INFER_TMP_BLOB_DESC(tmp);
  OF_HINGE_LOSS_INFER_TMP_BLOB_DESC(tmp_diff);
  OF_HINGE_LOSS_INFER_TMP_BLOB_DESC(tmp_storage);
#undef OF_HINGE_LOSS_INFER_TMP_BLOB_DESC
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kHingeLossConf, HingeLossOp);

}  // namespace oneflow
