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
#include "oneflow/core/kernel/new_kernel_util.h"

#ifdef OF_ENABLE_PROFILER
#include <nvtx3/nvToolsExt.h>
#endif  // OF_ENABLE_PROFILER

namespace oneflow {

namespace {

#ifdef OF_ENABLE_PROFILER
static thread_local HashMap<std::string, nvtxRangeId_t> mark2range_id;
#endif

}  // namespace

class NvtxOpKernelState final : public user_op::OpKernelState {
 public:
  NvtxOpKernelState() : counter_(0) {
#ifndef OF_ENABLE_PROFILER
    LOG(WARNING) << "To use NVTX, run cmake with -DBUILD_PROFILER=ON";
#endif
  }
  ~NvtxOpKernelState() override = default;

  int64_t counter() const { return counter_; }
  void IncreaseCount() { counter_ += 1; }

 private:
  int64_t counter_;
};

class NvtxStartKernel final : public user_op::OpKernel {
 public:
  NvtxStartKernel() = default;
  ~NvtxStartKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NvtxOpKernelState>();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape_view();
    CHECK_EQ(out->shape_view(), in_shape);
    const DataType in_data_type = in->data_type();
    CHECK_EQ(out->data_type(), in_data_type);
    Memcpy<DeviceType::kCUDA>(ctx->stream(), out->mut_dptr<void>(), in->dptr<void>(),
                              in_shape.elem_cnt() * GetSizeOfDataType(in_data_type));
#ifdef OF_ENABLE_PROFILER
    auto* kernel_state = dynamic_cast<NvtxOpKernelState*>(state);
    const std::string mark_prefix = ctx->Attr<std::string>("mark_prefix");
    const std::string mark = mark_prefix + "-" + std::to_string(kernel_state->counter());
    nvtxRangeId_t range_id = nvtxRangeStartA(mark.c_str());
    CHECK(mark2range_id.emplace(mark, range_id).second);
    kernel_state->IncreaseCount();
#endif  // OF_ENABLE_PROFILER
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("nvtx_start")
    .SetCreateFn<NvtxStartKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));
      return Maybe<void>::Ok();
    });

class NvtxEndKernel final : public user_op::OpKernel {
 public:
  NvtxEndKernel() = default;
  ~NvtxEndKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<NvtxOpKernelState>();
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape_view();
    CHECK_EQ(out->shape_view(), in_shape);
    const DataType in_data_type = in->data_type();
    CHECK_EQ(out->data_type(), in_data_type);
#ifdef OF_ENABLE_PROFILER
    auto* kernel_state = dynamic_cast<NvtxOpKernelState*>(state);
    const std::string mark_prefix = ctx->Attr<std::string>("mark_prefix");
    const std::string mark = mark_prefix + "-" + std::to_string(kernel_state->counter());
    auto it = mark2range_id.find(mark.c_str());
    CHECK(it != mark2range_id.end());
    nvtxRangeId_t range_id = it->second;
    mark2range_id.erase(it);
    nvtxRangeEnd(range_id);
    Memcpy<DeviceType::kCUDA>(ctx->stream(), out->mut_dptr<void>(), in->dptr<void>(),
                              in_shape.elem_cnt() * GetSizeOfDataType(in_data_type));
    kernel_state->IncreaseCount();
#endif
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("nvtx_end")
    .SetCreateFn<NvtxEndKernel>()
    .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCUDA)
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
