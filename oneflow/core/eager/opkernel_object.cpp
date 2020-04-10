#include "oneflow/core/eager/opkernel_object.h"

namespace oneflow {
namespace eager {

void OpKernelObject::ResetOpAndKernel(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  auto op = ConstructOp(op_conf_, device_type_, job_desc_.get());
  ParallelContext parallel_ctx;
  {
    parallel_ctx.set_parallel_id(0);
    parallel_ctx.set_parallel_num(1);
  }
  std::unique_ptr<OpContext> op_ctx;
  InferBlobDescs(*op, BlobDesc4BnInOp, &parallel_ctx, &op_ctx);
  NewPartialInitializedKernel(*op, BlobDesc4BnInOp, &parallel_ctx, op_ctx.get());
}

void OpKernelObject::InferBlobDescs(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, std::unique_ptr<OpContext>* op_ctx) {
  SbpSignature sbp_signature;
  {
    auto* map = sbp_signature.mutable_bn_in_op2sbp_parallel();
    for (const auto& ibn : op.input_bns()) { (*map)[ibn].mutable_broadcast_parallel(); }
    for (const auto& obn : op.output_bns()) { (*map)[obn].mutable_broadcast_parallel(); }
  }
  CHECK_JUST(op.InferBlobDescsIf(BlobDesc4BnInOp, parallel_ctx, &sbp_signature,
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

}  // namespace eager
}  // namespace oneflow
