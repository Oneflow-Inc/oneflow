#include "oneflow/core/operator/center_loss_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"

namespace oneflow {
  void CenterLossOp::VirtualInitFromOpConf() {
    EnrollForwardModelBn("centers");
    EnrollDataTmpBn("piece_centers");
  }

  const PbMessage& CenterLossOp::GetCustomizedConf() const {
    return op_conf().center_loss_conf();
  }

  LossKernelConf* CenterLossOp::GetMutLossKernelConf(
      KernelConf* kernel_conf) const {
    return kernel_conf->mutable_center_loss_conf()->mutable_loss_conf();
  }

  void CenterLossOp::VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const {
    const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
    CHECK_EQ(pred_blob_desc->shape().NumAxes(), 2);
    BlobDesc* piece_centers_blob_desc = GetBlobDesc4BnInOp("piece_centers");
    *piece_centers_blob_desc = *pred_blob_desc;
  }

  REGISTER_OP(OperatorConf::kCenterLossConf, CenterLossOp);
}