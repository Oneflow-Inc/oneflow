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
#include "oneflow/user/kernels/collective_communication/include/communication_context.h"
#include "oneflow/user/kernels/collective_communication/include/all_reduce.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

auto AllReduceCollectiveCommunicationExists() {
  return hob::make_custom("AllReduceCollectiveCommunicationExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            DeviceType device_type = ctx.device_type();
                            return ccl::IsCommunicationContextRegistered(device_type)
                                   && ccl::IsAllReduceRegistered(device_type);
                          });
}

class EagerCclOpKernelCache final : public user_op::OpKernelCache {
 public:
  explicit EagerCclOpKernelCache(user_op::KernelCacheContext* ctx) { Init(ctx); }
  ~EagerCclOpKernelCache() override = default;

  const std::shared_ptr<ccl::CommunicationContext>& communication_ctx() const {
    return communication_ctx_;
  }

 private:
  void Init(user_op::KernelCacheContext* ctx) {
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    ParallelConf parallel_conf;
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    Symbol<ParallelDesc> parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
    communication_ctx_ = ccl::NewCommunicationContext(parallel_desc->device_type(), parallel_desc);
  }

  std::shared_ptr<ccl::CommunicationContext> communication_ctx_;
};

void InitEagerCclOpKernelCache(user_op::KernelCacheContext* ctx,
                               std::shared_ptr<user_op::OpKernelCache>* cache_ptr) {
  // NOTE(jianhao): the cache only depends on parallel_conf, and the kernel is singleton
  // once parallel_conf is determined, so only init the cache at the first time.
  if (*cache_ptr == nullptr) { *cache_ptr = std::make_shared<EagerCclOpKernelCache>(ctx); }
}

}  // namespace

class EagerCclAllReduceKernel final : public user_op::OpKernel {
 public:
  EagerCclAllReduceKernel() = default;
  ~EagerCclAllReduceKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    InitEagerCclOpKernelCache(ctx, cache_ptr);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* kernel_cache = dynamic_cast<const EagerCclOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape_view(), out->shape_view()) << kOfBugIssueUploadPrompt;
    CHECK_EQ(in->data_type(), out->data_type()) << kOfBugIssueUploadPrompt;

    ccl::ReduceType reduce_type = ccl::kSum;
    if (in->data_type() == kBool) { reduce_type = ccl::kMax; }

    std::unique_ptr<ccl::AllReduce> all_reduce = ccl::NewCollectiveCommunication<ccl::AllReduce>(
        ctx->device_type(), in->data_type(), reduce_type);
    all_reduce->Launch(ctx->stream(), in->dptr(), out->mut_dptr(), out->shape_view().elem_cnt(),
                       kernel_cache->communication_ctx());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_ccl_all_reduce")
    .SetCreateFn<EagerCclAllReduceKernel>()
    .SetIsMatchedHob(AllReduceCollectiveCommunicationExists());

}  // namespace oneflow
