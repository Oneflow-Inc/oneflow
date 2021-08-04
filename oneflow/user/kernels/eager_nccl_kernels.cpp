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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

class EagerNcclOpKernelState final : public user_op::OpKernelState {
 public:
  EagerNcclOpKernelState(user_op::KernelInitContext* ctx) { Init(ctx); }
  ~EagerNcclOpKernelState() override = default;

  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }

 private:
  void Init(user_op::KernelInitContext* ctx) {
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    ParallelConf parallel_conf;
    std::set<std::pair<int64_t, int64_t>> device_set;
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    parallel_desc_ = SymbolOf(ParallelDesc(parallel_conf));
  }

  Symbol<ParallelDesc> parallel_desc_;
};

}  // namespace

class EagerNcclBroadcastKernel final : public user_op::OpKernel {
 public:
  EagerNcclBroadcastKernel() = default;
  ~EagerNcclBroadcastKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerNcclOpKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerNcclOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape(), out->shape());
    CHECK_EQ(in->data_type(), out->data_type());
    int64_t root = ctx->Attr<int64_t>("root");
    CHECK_JUST(ccl::Broadcast<DeviceType::kCPU>(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                                in->data_type(), root,
                                                kernel_state->parallel_desc(), ctx->device_ctx()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_broadcast")
    .SetCreateFn<EagerNcclBroadcastKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "cpu");

}  // namespace oneflow
