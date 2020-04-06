#include "oneflow/core/eager/opkernel_object.h"

namespace oneflow {
namespace eager {

void OpKernelObject::InferAndNewUninitiatedKernel(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  ParallelContext parallel_ctx;
  {
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(1);
  }
  InferBlobDescs(BlobDesc4BnInOp, &parallel_ctx);
  NewUninitiatedKernel(BlobDesc4BnInOp, &parallel_ctx);
}

void OpKernelObject::InferBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  SbpSignature sbp_signature;
  {
    auto* map = sbp_signature.mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : op_->input_bns()) { (*map)[ibn].mutable_broadcast_parallel(); }
    for (const auto& obn : op_->output_bns()) { (*map)[obn].mutable_broadcast_parallel(); }
  }
  CHECK_JUST(op_->InferBlobDescsIf(BlobDesc4BnInOp, parallel_ctx, &sbp_signature,
                                   [this](OpContext* op_ctx) { op_ctx_.reset(op_ctx); }));
}

void OpKernelObject::NewUninitiatedKernel(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) {
  auto LogicalBlobDesc4BnInOpFunc = [&](const std::string& bn_in_op) -> const BlobDesc& {
    return *BlobDesc4BnInOp(bn_in_op);
  };
  op_->GenKernelConf(BlobDesc4BnInOp, parallel_ctx, &kernel_conf_, op_ctx_.get(),
                     LogicalBlobDesc4BnInOpFunc);
  kernel_.reset(ConstructUninitiatedKernel(kernel_conf_));
}

const Kernel& OpKernelObject::kernel(const KernelCtx& ctx,
                                     const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
  if (!is_kernel_initiated_) {
    // malloc for const_buf blob and tmp blob
    kernel_->Init(job_desc_.get(), kernel_conf_, ctx.device_ctx);
    kernel_->InitModelAndConstBuf(ctx, BnInOp2Blob);
    is_kernel_initiated_ = true;
  }
  return *kernel_;
}

}  // namespace eager
}  // namespace oneflow
