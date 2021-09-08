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
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

namespace {

void Swap(int64_t& m, int64_t& n) {
  int64_t temp = m;
  m = n;
  n = temp;
}

int64_t GCD(int64_t m, int64_t n) {
  if (m < 0 || n < 0) { return 0; }
  if (m < n) { Swap(m, n); }
  if (n == 0) { return m; }
  return GCD(n, m % n);
}

int64_t LCM(int m, int n) {
  if (m < 0 || n < 0) { return 0; }
  return (m * n / GCD(m, n));
}

class EagerCclS0ToS0OpKernelState final : public user_op::OpKernelState {
 public:
  explicit EagerCclS0ToS0OpKernelState(user_op::KernelInitContext* ctx) : num_of_data_piece_(0) {
    Init(ctx);
  }
  ~EagerCclS0ToS0OpKernelState() override = default;

  Symbol<ParallelDesc> in_parallel_desc() const { return in_parallel_desc_; }
  Symbol<ParallelDesc> out_parallel_desc() const { return out_parallel_desc_; }

  int64_t num_of_data_piece() const { return num_of_data_piece_; }

  const HashMap<int64_t, std::pair<int64_t, int64_t>>& data_piece_idx_to_p2p_pair() const {
    return data_piece_idx_to_p2p_pair_;
  }

 private:
  void Init(user_op::KernelInitContext* ctx) {
    const std::string& in_parallel_conf_txt = ctx->Attr<std::string>("in_parallel_conf");
    const std::string& out_parallel_conf_txt = ctx->Attr<std::string>("out_parallel_conf");
    in_parallel_desc_ = CHECK_JUST(TxtStringToPlacement(in_parallel_conf_txt));
    out_parallel_desc_ = CHECK_JUST(TxtStringToPlacement(out_parallel_conf_txt));
    int64_t in_parallel_num = in_parallel_desc_->parallel_num();
    int64_t out_parallel_num = out_parallel_desc_->parallel_num();

    num_of_data_piece_ = LCM(in_parallel_num, out_parallel_num);
    for (int64_t i = 0; i < num_of_data_piece_; ++i) {
      int64_t src_parallel_id = i / (num_of_data_piece_ / in_parallel_num);
      int64_t dst_parallel_id = i / (num_of_data_piece_ / out_parallel_num);
      int64_t src = CHECK_JUST(in_parallel_desc_->MachineId4ParallelId(src_parallel_id));
      int64_t dst = CHECK_JUST(out_parallel_desc_->MachineId4ParallelId(dst_parallel_id));
      CHECK(data_piece_idx_to_p2p_pair_.emplace(i, std::make_pair(src, dst)).second);
    }
  }

  Symbol<ParallelDesc> in_parallel_desc_;
  Symbol<ParallelDesc> out_parallel_desc_;
  int64_t num_of_data_piece_;
  HashMap<int64_t, std::pair<int64_t, int64_t>> data_piece_idx_to_p2p_pair_;
};

}  // namespace

class EagerCclS0ToS0Kernel final : public user_op::OpKernel {
 public:
  EagerCclS0ToS0Kernel() = default;
  ~EagerCclS0ToS0Kernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerCclS0ToS0OpKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerCclS0ToS0OpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const void* in_ptr = in->dptr();
    void* out_ptr = out->mut_dptr();

    const int64_t total_elem_cnt = ctx->Attr<Shape>("shape").elem_cnt();
    int64_t num_of_data_piece = kernel_state->num_of_data_piece();
    const int64_t elem_per_data_piece = total_elem_cnt / num_of_data_piece;
    const int64_t size_per_data_piece = elem_per_data_piece * GetSizeOfDataType(in->data_type());

    const auto& data_piece_idx_to_p2p_pair = kernel_state->data_piece_idx_to_p2p_pair();

    for (const auto& elem : data_piece_idx_to_p2p_pair) {
      int64_t data_piece_idx = elem.first;
      int64_t src = elem.second.first;
      int64_t dst = elem.second.second;

      if (GlobalProcessCtx::Rank() == src) {
        Symbol<ParallelDesc> in_parallel_desc = kernel_state->in_parallel_desc();
        int64_t device_id = GlobalProcessCtx::LocalRank(src);
        int64_t in_parallel_id =
            CHECK_JUST(in_parallel_desc->ParallelId4MachineDeviceId(src, device_id));
        int64_t in_parallel_num = in_parallel_desc->parallel_num();

        int64_t src_idx_offset =
            data_piece_idx - (in_parallel_id * num_of_data_piece / in_parallel_num);

        CHECK_JUST(Send<DeviceType::kCPU>(
            reinterpret_cast<const void*>(reinterpret_cast<const char*>(in_ptr)
                                          + src_idx_offset * size_per_data_piece),
            elem_per_data_piece, in->data_type(), dst, ctx->device_ctx()));
      }
      if (GlobalProcessCtx::Rank() == dst) {
        Symbol<ParallelDesc> out_parallel_desc = kernel_state->out_parallel_desc();
        int64_t device_id = GlobalProcessCtx::LocalRank(dst);
        int64_t out_parallel_id =
            CHECK_JUST(out_parallel_desc->ParallelId4MachineDeviceId(dst, device_id));
        int64_t out_parallel_num = out_parallel_desc->parallel_num();

        int64_t dst_idx_offset =
            data_piece_idx - (out_parallel_id * num_of_data_piece / out_parallel_num);

        CHECK_JUST(
            Recv<DeviceType::kCPU>(reinterpret_cast<void*>(reinterpret_cast<char*>(out_ptr)
                                                           + dst_idx_offset * size_per_data_piece),
                                   elem_per_data_piece, out->data_type(), src, ctx->device_ctx()));
      }
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_s0_to_s0")
    .SetCreateFn<EagerCclS0ToS0Kernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "cpu");

}  // namespace oneflow
