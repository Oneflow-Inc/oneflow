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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/placement_sbp_util.h"

namespace oneflow {

namespace {

class EagerSToBOpKernelState final : public user_op::OpKernelState {
 public:
  explicit EagerSToBOpKernelState(user_op::KernelInitContext* ctx) : in_parallel_num_(0) {
    Init(ctx);
  }
  ~EagerSToBOpKernelState() override = default;

  int64_t in_parallel_num() const { return in_parallel_num_; }

  const HashMap<std::pair<int64_t, int64_t>, int64_t>& p2p_pair_to_in_parallel_id() const {
    return p2p_pair_to_in_parallel_id_;
  }

 private:
  void Init(user_op::KernelInitContext* ctx) {
    const std::string& in_parallel_conf_txt = ctx->Attr<std::string>("in_parallel_conf");
    const std::string& out_parallel_conf_txt = ctx->Attr<std::string>("out_parallel_conf");
    Symbol<ParallelDesc> in_parallel_desc = CHECK_JUST(TxtStringToPlacement(in_parallel_conf_txt));
    Symbol<ParallelDesc> out_parallel_desc =
        CHECK_JUST(TxtStringToPlacement(out_parallel_conf_txt));
    int64_t out_parallel_num = out_parallel_desc->parallel_num();
    in_parallel_num_ = in_parallel_desc->parallel_num();

    for (int64_t out_parallel_id = 0; out_parallel_id < out_parallel_num; ++out_parallel_id) {
      int64_t dst = CHECK_JUST(out_parallel_desc->MachineId4ParallelId(out_parallel_id));
      for (int64_t in_parallel_id = 0; in_parallel_id < in_parallel_num_; ++in_parallel_id) {
        int64_t src = CHECK_JUST(in_parallel_desc->MachineId4ParallelId(in_parallel_id));
        CHECK(p2p_pair_to_in_parallel_id_.emplace(std::make_pair(src, dst), in_parallel_id).second);
      }
    }
  }

  int64_t in_parallel_num_;
  HashMap<std::pair<int64_t, int64_t>, int64_t> p2p_pair_to_in_parallel_id_;
};

}  // namespace

template<DeviceType device_type>
class EagerSToBKernel final : public user_op::OpKernel {
 public:
  EagerSToBKernel() = default;
  ~EagerSToBKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerSToBOpKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerSToBOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const void* in_ptr = in->dptr();
    void* out_ptr = out->mut_dptr();

    const int64_t total_elem_cnt = ctx->Attr<Shape>("shape").elem_cnt();
    int64_t in_parallel_num = kernel_state->in_parallel_num();
    const int64_t elem_per_data_piece = total_elem_cnt / in_parallel_num;
    const int64_t size_per_data_piece = elem_per_data_piece * GetSizeOfDataType(in->data_type());

    const auto& p2p_pair_to_in_parallel_id = kernel_state->p2p_pair_to_in_parallel_id();

    for (const auto& elem : p2p_pair_to_in_parallel_id) {
      int64_t in_parallel_id = elem.second;
      int64_t src = elem.first.first;
      int64_t dst = elem.first.second;

      if (GlobalProcessCtx::Rank() == src) {
        CHECK_JUST(Send<device_type>(in_ptr, elem_per_data_piece, in->data_type(), dst,
                                     ctx->device_ctx()));
      }
      if (GlobalProcessCtx::Rank() == dst) {
        CHECK_JUST(
            Recv<device_type>(reinterpret_cast<void*>(reinterpret_cast<char*>(out_ptr)
                                                      + in_parallel_id * size_per_data_piece),
                              elem_per_data_piece, out->data_type(), src, ctx->device_ctx()));
      }
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EAGER_B_TO_S_KERNEL(device)  \
  REGISTER_USER_KERNEL("eager_s_to_b")        \
      .SetCreateFn<EagerSToBKernel<device>>() \
      .SetIsMatchedHob(user_op::HobDeviceTag() == device);

REGISTER_EAGER_B_TO_S_KERNEL(DeviceType::kCPU)
#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
REGISTER_EAGER_B_TO_S_KERNEL(DeviceType::kGPU)
#endif

}  // namespace oneflow
