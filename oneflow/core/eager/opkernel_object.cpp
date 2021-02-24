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
    const OpNodeSignatureDesc& op_node_signature, const ParallelContext* parallel_ctx,
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc* parallel_desc) {
  auto op = ConstructOp(op_conf_, device_type_, job_desc_.get());
  JUST(op->FillOpParallelDesc(*parallel_desc));
  const auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
    return CHECK_JUST(op_node_signature.LogicalBlobDesc4BnInOp(bn));
  };
  JUST(op->FillLogicalInBlobDesc(LogicalBlobDesc4BnInOp));
  JUST(op->FillLogicalOutBlobDesc(LogicalBlobDesc4BnInOp));
  JUST(op->FillSbpSignature(op_node_signature.sbp_signature()));
  JUST(InferBlobDescs(*op, BlobDesc4BnInOp, &op_node_signature.sbp_signature(), parallel_ctx));
  NewPartialInitializedKernel(*op, BlobDesc4BnInOp, op_node_signature, parallel_ctx, parallel_desc);
  return Maybe<void>::Ok();
}

Maybe<void> OpKernelObject::InferBlobDescs(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx) {
  JUST(op.InferBlobDescsIf(BlobDesc4BnInOp, parallel_ctx, sbp_signature));
  return Maybe<void>::Ok();
}

void OpKernelObject::NewPartialInitializedKernel(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const OpNodeSignatureDesc& op_node_signature, const ParallelContext* parallel_ctx,
    const ParallelDesc* parallel_desc) {
  KernelConf kernel_conf;
  op.GenKernelConf(BlobDesc4BnInOp, parallel_ctx, &kernel_conf);
  kernel_.reset(new EagerKernel(job_desc_.get(), kernel_conf));
}

Maybe<void> SystemOpKernelObject::ResetKernel(
    const OpNodeSignatureDesc& op_node_signature, const ParallelContext* parallel_ctx,
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc* parallel_desc) {
  auto op = ConstructOp(op_conf_, device_type_, job_desc_.get());
  JUST(op->FillOpParallelDesc(*parallel_desc));
  const auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
    return CHECK_JUST(op_node_signature.LogicalBlobDesc4BnInOp(bn));
  };
  JUST(op->FillLogicalInBlobDesc(LogicalBlobDesc4BnInOp));
  JUST(op->FillLogicalOutBlobDesc(LogicalBlobDesc4BnInOp));
  JUST(op->FillSbpSignature(op_node_signature.sbp_signature()));
  JUST(InferBlobDescs(*op, BlobDesc4BnInOp, &op_node_signature.sbp_signature(), parallel_ctx));
  ResetKernel(*op, BlobDesc4BnInOp, op_node_signature, parallel_ctx, parallel_desc);
  return Maybe<void>::Ok();
}

Maybe<void> SystemOpKernelObject::InferBlobDescs(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const SbpSignature* sbp_signature, const ParallelContext* parallel_ctx) {
  JUST(op.InferBlobDescsIf(BlobDesc4BnInOp, parallel_ctx, sbp_signature));
  return Maybe<void>::Ok();
}

void SystemOpKernelObject::ResetKernel(
    const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const OpNodeSignatureDesc& op_node_signature, const ParallelContext* parallel_ctx,
    const ParallelDesc* parallel_desc) {
  KernelConf kernel_conf;
  op.GenKernelConf(BlobDesc4BnInOp, parallel_ctx, &kernel_conf);
  kernel_ = ConstructKernel(job_desc_.get(), kernel_conf, nullptr);
}

}  // namespace eager
}  // namespace oneflow
