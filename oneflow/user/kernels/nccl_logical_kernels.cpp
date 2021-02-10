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
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

class NcclLogicalKernelCommState final : public user_op::OpKernelState {
 public:
  NcclLogicalKernelCommState(user_op::KernelInitContext* ctx) {
    std::set<std::pair<int64_t, int64_t>> device_set;
    const ParallelDesc& parallel_desc = ctx->parallel_desc();
    FOR_RANGE(int64_t, parallel_id, 0, parallel_desc.parallel_num()) {
      int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    comm_ = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set);
  }
  ~NcclLogicalKernelCommState() = default;

  ncclComm_t comm() const { return comm_; }

 private:
  ncclComm_t comm_;
};

class NcclLogicalAllReduceKernel final : public user_op::OpKernel {
 public:
  NcclLogicalAllReduceKernel() = default;
  ~NcclLogicalAllReduceKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogicalKernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogicalKernelCommState*>(state);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape(), out->shape());
    CHECK_EQ(in->data_type(), out->data_type());
    OF_NCCL_CHECK(ncclAllReduce(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class NcclLogicalReduceScatterKernel final : public user_op::OpKernel {
 public:
  NcclLogicalReduceScatterKernel() = default;
  ~NcclLogicalReduceScatterKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogicalKernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogicalKernelCommState*>(state);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = ctx->parallel_ctx().parallel_num();
    CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt() * num_ranks);
    OF_NCCL_CHECK(ncclReduceScatter(in->dptr(), out->mut_dptr(), out->shape().elem_cnt(),
                                    GetNcclDataType(in->data_type()), ncclRedOp_t::ncclSum,
                                    nccl_comm->comm(), ctx->device_ctx()->cuda_stream()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

REGISTER_USER_KERNEL("_nccl_logical_op_all_reduce")
    .SetCreateFn<NcclLogicalAllReduceKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

REGISTER_USER_KERNEL("_nccl_logical_op_reduce_scatter")
    .SetCreateFn<NcclLogicalReduceScatterKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

}  // namespace oneflow
