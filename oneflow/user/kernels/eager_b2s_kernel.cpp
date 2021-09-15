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
#include "oneflow/user/kernels/communicate_util.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/placement_sbp_util.h"

namespace oneflow {

namespace {

class EagerBToSOpKernelState final : public user_op::OpKernelState {
 public:
  explicit EagerBToSOpKernelState(user_op::KernelInitContext* ctx) : out_parallel_num_(0) {
    Init(ctx);
  }
  ~EagerBToSOpKernelState() override = default;

  int64_t out_parallel_num() const { return out_parallel_num_; }

  const HashMap<int64_t, std::pair<int64_t, int64_t>>& out_parallel_id_to_p2p_pair() const {
    return out_parallel_id_to_p2p_pair_;
  }

 private:
  void Init(user_op::KernelInitContext* ctx) {
    const std::string& in_parallel_conf_txt = ctx->Attr<std::string>("in_parallel_conf");
    const std::string& out_parallel_conf_txt = ctx->Attr<std::string>("out_parallel_conf");
    Symbol<ParallelDesc> in_parallel_desc = CHECK_JUST(TxtStringToPlacement(in_parallel_conf_txt));
    Symbol<ParallelDesc> out_parallel_desc =
        CHECK_JUST(TxtStringToPlacement(out_parallel_conf_txt));
    out_parallel_num_ = out_parallel_desc->parallel_num();

    for (int64_t out_parallel_id = 0; out_parallel_id < out_parallel_num_; ++out_parallel_id) {
      int64_t dst = CHECK_JUST(out_parallel_desc->MachineId4ParallelId(out_parallel_id));
      int64_t src = -1;
      if (in_parallel_desc->ContainingMachineId(dst)) {
        src = dst;
      } else {
        int64_t in_parallel_num = in_parallel_desc->parallel_num();
        int64_t src_parallel_id = out_parallel_id % in_parallel_num;
        src = CHECK_JUST(in_parallel_desc->MachineId4ParallelId(src_parallel_id));
      }
      CHECK_NE(src, -1);
      CHECK(out_parallel_id_to_p2p_pair_.emplace(out_parallel_id, std::make_pair(src, dst)).second);
    }
  }

  int64_t out_parallel_num_;
  HashMap<int64_t, std::pair<int64_t, int64_t>> out_parallel_id_to_p2p_pair_;
};

}  // namespace

template<DeviceType device_type>
class EagerBToSKernel final : public user_op::OpKernel {
 public:
  EagerBToSKernel() = default;
  ~EagerBToSKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerBToSOpKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerBToSOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const void* in_ptr = in->dptr();
    void* out_ptr = out->mut_dptr();

    const int64_t total_elem_cnt = ctx->Attr<Shape>("shape").elem_cnt();
    int64_t out_parallel_num = kernel_state->out_parallel_num();
    const int64_t elem_cnt_per_rank = total_elem_cnt / out_parallel_num;
    const int64_t data_size_per_rank = elem_cnt_per_rank * GetSizeOfDataType(in->data_type());

    const auto& out_parallel_id_to_p2p_pair = kernel_state->out_parallel_id_to_p2p_pair();

    for (const auto& elem : out_parallel_id_to_p2p_pair) {
      int64_t out_parallel_id = elem.first;
      int64_t src = elem.second.first;
      int64_t dst = elem.second.second;

      if (GlobalProcessCtx::Rank() == src) {
        CHECK_JUST(
            Send<device_type>(reinterpret_cast<const void*>(reinterpret_cast<const char*>(in_ptr)
                                                            + out_parallel_id * data_size_per_rank),
                              elem_cnt_per_rank, in->data_type(), dst, ctx->device_ctx()));
      }
      if (GlobalProcessCtx::Rank() == dst) {
        CHECK_JUST(Recv<device_type>(out_ptr, elem_cnt_per_rank, out->data_type(), src,
                                     ctx->device_ctx()));
      }
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EAGER_B_TO_S_KERNEL(device)  \
  REGISTER_USER_KERNEL("eager_b_to_s")        \
      .SetCreateFn<EagerBToSKernel<device>>() \
      .SetIsMatchedHob(user_op::HobDeviceTag() == device);

REGISTER_EAGER_B_TO_S_KERNEL(DeviceType::kCPU)
#if defined(WITH_CUDA) && HAS_GPU_SEND_RECV
REGISTER_EAGER_B_TO_S_KERNEL(DeviceType::kGPU)
#endif

}  // namespace oneflow
