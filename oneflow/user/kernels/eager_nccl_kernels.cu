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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

class EagerNcclOpKernelState final : public user_op::OpKernelState {
 public:
  EagerNcclOpKernelState(user_op::KernelInitContext* ctx) { Init(ctx); }
  ~EagerNcclOpKernelState() override = default;

  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }
  ncclComm_t comm() const { return comm_; }

 private:
  void Init(user_op::KernelInitContext* ctx) {
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    ParallelConf parallel_conf;
    std::set<std::pair<int64_t, int64_t>> device_set;
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    parallel_desc_ = SymbolOf(ParallelDesc(parallel_conf));
    FOR_RANGE(int64_t, parallel_id, 0, parallel_desc_->parallel_num()) {
      int64_t machine_id = CHECK_JUST(parallel_desc_->MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc_->DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    comm_ = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set);
  }

  Symbol<ParallelDesc> parallel_desc_;
  ncclComm_t comm_;
};

}  // namespace

class EagerNcclAllReduceKernel final : public user_op::OpKernel {
 public:
  EagerNcclAllReduceKernel() = default;
  ~EagerNcclAllReduceKernel() override = default;

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
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), ncclSum, kernel_state->comm(),
                                ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_all_reduce")
    .SetCreateFn<EagerNcclAllReduceKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

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
    int64_t root = ctx->Attr<int64_t>("root");
    const void* in_ptr = nullptr;
    if (GlobalProcessCtx::Rank() == root) {
      CHECK_EQ(in->shape(), out->shape());
      CHECK_EQ(in->data_type(), out->data_type());
      in_ptr = in->dptr();
    }
    OF_NCCL_CHECK(ncclBroadcast(in_ptr, out->mut_dptr(), out->shape().elem_cnt(),
                                GetNcclDataType(out->data_type()), root, kernel_state->comm(),
                                ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_broadcast")
    .SetCreateFn<EagerNcclBroadcastKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

class EagerNcclReduceKernel final : public user_op::OpKernel {
 public:
  EagerNcclReduceKernel() = default;
  ~EagerNcclReduceKernel() override = default;

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
    OF_NCCL_CHECK(ncclReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                             GetNcclDataType(in->data_type()), ncclSum, root, kernel_state->comm(),
                             ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_reduce")
    .SetCreateFn<EagerNcclReduceKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

class EagerNcclReduceScatterKernel final : public user_op::OpKernel {
 public:
  EagerNcclReduceScatterKernel() = default;
  ~EagerNcclReduceScatterKernel() override = default;

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
    CHECK_EQ(in->data_type(), out->data_type());
    const auto& op_type = ctx->Attr<std::string>("op_type");
    OF_NCCL_CHECK(ncclReduceScatter(in->dptr(), out->mut_dptr(), out->shape().elem_cnt(),
                                    GetNcclDataType(in->data_type()),
                                    CHECK_JUST(MapAt(op_type2ncclRedOp_t, op_type)),
                                    kernel_state->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  static HashMap<std::string, ncclRedOp_t> op_type2ncclRedOp_t;
};

HashMap<std::string, ncclRedOp_t> EagerNcclReduceScatterKernel::op_type2ncclRedOp_t = {
    {"sum", ncclSum}, {"max", ncclMax}};

REGISTER_USER_KERNEL("eager_nccl_reduce_scatter")
    .SetCreateFn<EagerNcclReduceScatterKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

class EagerNcclAllGatherKernel final : public user_op::OpKernel {
 public:
  EagerNcclAllGatherKernel() = default;
  ~EagerNcclAllGatherKernel() override = default;

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
    CHECK_EQ(in->data_type(), out->data_type());
    OF_NCCL_CHECK(ncclAllGather(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), kernel_state->comm(),
                                ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_all_gather")
    .SetCreateFn<EagerNcclAllGatherKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");
}  // namespace oneflow
