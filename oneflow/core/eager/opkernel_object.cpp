#include "oneflow/core/eager/opkernel_object.h"

namespace oneflow {
namespace eager {

void OpKernelObject::ResetOpAndKernel(
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx,
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  auto op = ConstructOp(op_conf_, device_type_, job_desc_.get());
  std::unique_ptr<OpContext> op_ctx;
  InferBlobDescs(*op, BlobDesc4BnInOp, sbp_signature, parallel_ctx, &op_ctx);
  NewPartialInitializedKernel(*op, BlobDesc4BnInOp, parallel_ctx, op_ctx.get());
}

void OpKernelObject::InferBlobDescs(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx,
    std::unique_ptr<OpContext>* op_ctx) {
  CHECK_JUST(op.InferBlobDescsIf(BlobDesc4BnInOp, parallel_ctx, sbp_signature,
                                 [op_ctx](OpContext* ctx) { op_ctx->reset(ctx); }));
}

void OpKernelObject::NewPartialInitializedKernel(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, OpContext* op_ctx) {
  KernelConf kernel_conf;
  auto LogicalBlobDesc4BnInOpFunc = [&](const std::string& bn_in_op) -> const BlobDesc& {
    return *BlobDesc4BnInOp(bn_in_op);
  };
  op.GenKernelConf(BlobDesc4BnInOp, parallel_ctx, &kernel_conf, op_ctx, LogicalBlobDesc4BnInOpFunc);
  kernel_.reset(new EagerKernel(job_desc_.get(), kernel_conf));
}

void SystemOpKernelObject::ResetKernel(
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx,
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  auto op = ConstructOp(op_conf_, device_type_, job_desc_.get());
  std::unique_ptr<OpContext> op_ctx;
  InferBlobDescs(*op, BlobDesc4BnInOp, sbp_signature, parallel_ctx, &op_ctx);
  ResetKernel(*op, BlobDesc4BnInOp, parallel_ctx, op_ctx.get());
}

void SystemOpKernelObject::InferBlobDescs(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx,
    std::unique_ptr<OpContext>* op_ctx) {
  CHECK_JUST(op.InferBlobDescsIf(BlobDesc4BnInOp, parallel_ctx, sbp_signature,
                                 [op_ctx](OpContext* ctx) { op_ctx->reset(ctx); }));
}

void SystemOpKernelObject::ResetKernel(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, OpContext* op_ctx) {
  KernelConf kernel_conf;
  auto LogicalBlobDesc4BnInOpFunc = [&](const std::string& bn_in_op) -> const BlobDesc& {
    return *BlobDesc4BnInOp(bn_in_op);
  };
  op.GenKernelConf(BlobDesc4BnInOp, parallel_ctx, &kernel_conf, op_ctx, LogicalBlobDesc4BnInOpFunc);
  kernel_ = ConstructKernel(job_desc_.get(), kernel_conf, nullptr);
}

}  // namespace eager
}  // namespace oneflow
