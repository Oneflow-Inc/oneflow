#include "oneflow/core/operator/copy_hd_op.h"

namespace oneflow {

void CopyHdOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollOutputBn("out", false);
}

const PbMessage& CopyHdOp::GetCustomizedConf() const { return op_conf().copy_hd_conf(); }

void CopyHdOp::set_enable_synthetic_data(bool val) { enable_synthetic_data_ = val; }

void CopyHdOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  CopyHdKernelConf* conf = kernel_conf->mutable_copy_hd_conf();
  conf->set_enable_synthetic_data(enable_synthetic_data_);
}

REGISTER_OP(OperatorConf::kCopyHdConf, CopyHdOp);

}  // namespace oneflow
