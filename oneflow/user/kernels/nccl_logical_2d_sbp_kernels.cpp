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
#include "oneflow/core/kernel/new_kernel_util.h"

#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700

namespace oneflow {

namespace {

class NcclLogical2DSameDim0KernelCommState final : public user_op::OpKernelState {
 public:
  NcclLogical2DSameDim0KernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()) {}
  ~NcclLogical2DSameDim0KernelCommState() = default;

  ncclComm_t comm() {
    if (!is_init_) {
      std::set<std::pair<int64_t, int64_t>> device_set;
      const Shape& hierarchy = *parallel_desc_.hierarchy();
      CHECK_EQ(hierarchy.NumAxes(), 2);
      const int64_t num_groups = hierarchy.At(0);
      const int64_t group_size = hierarchy.At(1);
      CHECK_EQ(num_groups * group_size, parallel_desc_.parallel_num());
      const int64_t this_group_begin_parallel_id = this_parallel_id_ / group_size * group_size;
      CHECK_EQ(this_group_begin_parallel_id % group_size, 0);
      CHECK_LE(this_group_begin_parallel_id + group_size, parallel_desc_.parallel_num());
      for (int64_t id_in_group = 0; id_in_group < group_size; ++id_in_group) {
        const int64_t parallel_id = this_group_begin_parallel_id + id_in_group;
        const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
        const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
        device_set.emplace(std::make_pair(machine_id, device_id));
      }
      comm_ = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set);
      is_init_ = true;
    }
    return comm_;
  }

 private:
  bool is_init_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  ncclComm_t comm_;
};

class NcclLogical2DSameDim0AllReduce final : public user_op::OpKernel {
 public:
  NcclLogical2DSameDim0AllReduce() = default;
  ~NcclLogical2DSameDim0AllReduce() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical2DSameDim0KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical2DSameDim0KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
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

class NcclLogical2DSameDim1KernelCommState final : public user_op::OpKernelState {
 public:
  NcclLogical2DSameDim1KernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()) {}
  ~NcclLogical2DSameDim1KernelCommState() = default;

  ncclComm_t comm() {
    if (!is_init_) {
      std::set<std::pair<int64_t, int64_t>> device_set;
      const Shape& hierarchy = *parallel_desc_.hierarchy();
      CHECK_EQ(hierarchy.NumAxes(), 2);
      const int64_t group_size = hierarchy.At(0);
      const int64_t num_groups = hierarchy.At(1);
      CHECK_EQ(num_groups * group_size, parallel_desc_.parallel_num());
      const int64_t this_group_begin_parallel_id = this_parallel_id_ % num_groups;
      CHECK_LT(this_group_begin_parallel_id + (group_size - 1) * num_groups,
               parallel_desc_.parallel_num());
      for (int64_t id_in_group = 0; id_in_group < group_size; ++id_in_group) {
        const int64_t parallel_id = this_group_begin_parallel_id + (id_in_group * num_groups);
        const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
        const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
        device_set.emplace(std::make_pair(machine_id, device_id));
      }
      comm_ = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get())->GetCommForDevice(device_set);
      is_init_ = true;
    }
    return comm_;
  }

 private:
  bool is_init_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  ncclComm_t comm_;
};

class NcclLogical2DSameDim1AllReduce final : public user_op::OpKernel {
 public:
  NcclLogical2DSameDim1AllReduce() = default;
  ~NcclLogical2DSameDim1AllReduce() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NcclLogical2DSameDim1KernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* nccl_comm = dynamic_cast<NcclLogical2DSameDim1KernelCommState*>(state);
    CHECK(nccl_comm != nullptr);
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

}  // namespace

REGISTER_USER_KERNEL("_nccl_logical_2D_same_dim0_all_reduce")
    .SetCreateFn<NcclLogical2DSameDim0AllReduce>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

REGISTER_USER_KERNEL("_nccl_logical_2D_same_dim1_all_reduce")
    .SetCreateFn<NcclLogical2DSameDim0AllReduce>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu");

}  // namespace oneflow

#endif  // WITH_CUDA && NCCL_VERSION_CODE > 2700
