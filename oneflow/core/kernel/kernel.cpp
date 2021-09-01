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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/runtime_blob_shape_infer_helper.h"
#include "oneflow/core/kernel/kernel_observer.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

namespace {

bool IsAllBlobEmpty(const PbRpf<std::string>& bns, const KernelContext* ctx) {
  for (const auto& bn : bns) {
    Blob* blob = ctx->BnInOp2Blob(bn);
    if (blob && !blob->IsBodyEmpty()) { return false; }
  }
  return true;
}

}  // namespace

Kernel::Kernel() = default;

Kernel::~Kernel() = default;

void Kernel::InitBase(const JobDesc* job_desc, const KernelConf& kernel_conf) {
  if (shape_infer_helper_) { return; }
  kernel_conf_ = kernel_conf;
  shape_infer_helper_.reset(
      new RuntimeBlobShapeInferHelper(this->op_conf(), this->kernel_conf(), job_desc));
}

void Kernel::Init(const KernelConf& kernel_conf, KernelContext* ctx) {
  InitBase(ctx->job_desc(), kernel_conf);
  VirtualKernelInit(ctx);
}

void Kernel::DestroyState(void* state) const { CHECK(state == nullptr); }

void Kernel::Launch(const KernelContext* ctx) const {
  Global<KernelObserver>::Get()->WillForward(ctx, this);
  Forward(ctx);
  Global<KernelObserver>::Get()->DidForward(ctx, this);
}

void Kernel::Forward(const KernelContext* ctx) const {
  Global<KernelObserver>::Get()->WillForwardHeader(ctx, this);
  ForwardHeader(ctx);
  Global<KernelObserver>::Get()->DidForwardHeader(ctx, this);
  if ((!kernel_conf_.all_blobs_are_static()) && IsAllBlobEmpty(op_attribute().output_bns(), ctx)
      && IsStateless()) {
    return;
  }
  Global<KernelObserver>::Get()->WillForwardDataContent(ctx, this);
  ForwardDataContent(ctx);
  Global<KernelObserver>::Get()->DidForwardDataContent(ctx, this);
}

void Kernel::ForwardHeader(const KernelContext* ctx) const {
  if (!kernel_conf_.all_blobs_are_static()) { ForwardShape(ctx); }
}

void Kernel::ForwardShape(const KernelContext* ctx) const {
  return shape_infer_helper_->InferShape(
      [ctx](const std::string& bn) { return ctx->BnInOp2Blob(bn); });
}

std::unique_ptr<const Kernel> ConstructKernel(const KernelConf& conf, KernelContext* kernel_ctx) {
  auto op_type = conf.op_attribute().op_conf().op_type_case();
  CHECK_NE(op_type, OperatorConf::OpTypeCase::OP_TYPE_NOT_SET)
      << " ERROR! KernelConf: " << conf.DebugString() << " has NOT set op_type_case";
  Kernel* rptr = kernel_registration::CreateKernel(conf);
  if (rptr == nullptr) { rptr = NewObj<int32_t, Kernel>(op_type, conf); }
  CHECK_NOTNULL(rptr);
  rptr->Init(conf, kernel_ctx);
  return std::unique_ptr<const Kernel>(rptr);
}

}  // namespace oneflow
