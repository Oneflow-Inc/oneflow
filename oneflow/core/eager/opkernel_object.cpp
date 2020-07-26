/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/eager/opkernel_object.h"

namespace oneflow {
namespace eager {

Maybe<void> OpKernelObject::ResetOpAndKernel(
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx,
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  auto op = ConstructOp(op_conf_, device_type_, job_desc_.get());
  std::unique_ptr<OpContext> op_ctx;
  JUST(InferBlobDescs(*op, BlobDesc4BnInOp, sbp_signature, parallel_ctx, &op_ctx));
  NewPartialInitializedKernel(*op, BlobDesc4BnInOp, parallel_ctx, op_ctx.get());
  return Maybe<void>::Ok();
}

Maybe<void> OpKernelObject::InferBlobDescs(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx,
    std::unique_ptr<OpContext>* op_ctx) {
  JUST(op.InferBlobDescsIf(BlobDesc4BnInOp, parallel_ctx, sbp_signature,
                           [op_ctx](OpContext* ctx) { op_ctx->reset(ctx); }));
  return Maybe<void>::Ok();
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

Maybe<void> SystemOpKernelObject::ResetKernel(
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx,
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  auto op = ConstructOp(op_conf_, device_type_, job_desc_.get());
  std::unique_ptr<OpContext> op_ctx;
  JUST(InferBlobDescs(*op, BlobDesc4BnInOp, sbp_signature, parallel_ctx, &op_ctx));
  ResetKernel(*op, BlobDesc4BnInOp, parallel_ctx, op_ctx.get());
  return Maybe<void>::Ok();
}

Maybe<void> SystemOpKernelObject::InferBlobDescs(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx,
    std::unique_ptr<OpContext>* op_ctx) {
  JUST(op.InferBlobDescsIf(BlobDesc4BnInOp, parallel_ctx, sbp_signature,
                           [op_ctx](OpContext* ctx) { op_ctx->reset(ctx); }));
  return Maybe<void>::Ok();
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
