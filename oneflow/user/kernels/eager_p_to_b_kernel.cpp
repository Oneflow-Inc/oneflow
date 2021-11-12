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
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/framework/placement_sbp_util.h"
#include "oneflow/core/ep/include/primitive/add.h"

namespace oneflow {

namespace {

class EagerPToBOpKernelState final : public user_op::OpKernelState {
 public:
  explicit EagerPToBOpKernelState(user_op::KernelInitContext* ctx) { Init(ctx); }
  ~EagerPToBOpKernelState() override = default;

  const std::vector<std::pair<int64_t, int64_t>>& p2p_pair() const { return p2p_pair_; }

 private:
  void Init(user_op::KernelInitContext* ctx) {
    const std::string& in_parallel_conf_txt = ctx->Attr<std::string>("in_parallel_conf");
    const std::string& out_parallel_conf_txt = ctx->Attr<std::string>("out_parallel_conf");
    Symbol<ParallelDesc> in_parallel_desc = CHECK_JUST(TxtStringToPlacement(in_parallel_conf_txt));
    Symbol<ParallelDesc> out_parallel_desc =
        CHECK_JUST(TxtStringToPlacement(out_parallel_conf_txt));
    int64_t out_parallel_num = out_parallel_desc->parallel_num();
    int64_t in_parallel_num = in_parallel_desc->parallel_num();

    for (int64_t out_parallel_id = 0; out_parallel_id < out_parallel_num; ++out_parallel_id) {
      int64_t dst = CHECK_JUST(out_parallel_desc->MachineId4ParallelId(out_parallel_id));
      for (int64_t in_parallel_id = 0; in_parallel_id < in_parallel_num; ++in_parallel_id) {
        int64_t src = CHECK_JUST(in_parallel_desc->MachineId4ParallelId(in_parallel_id));
        p2p_pair_.emplace_back(std::make_pair(src, dst));
      }
    }
  }

  std::vector<std::pair<int64_t, int64_t>> p2p_pair_;
};

size_t InferEagerPToBKernelTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  const Shape& shape = ctx->Attr<Shape>("shape");
  size_t tensor_byte_size = shape.elem_cnt() * GetSizeOfDataType(in_tensor.data_type());
  return tensor_byte_size;
}

}  // namespace

template<DeviceType device_type>
class EagerPToBKernel final : public user_op::OpKernel {
 public:
  EagerPToBKernel() = default;
  ~EagerPToBKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerPToBOpKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerPToBOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const void* in_ptr = in->dptr();
    void* tmp_buffer_ptr = tmp_buffer->mut_dptr();

    const int64_t total_elem_cnt = ctx->Attr<Shape>("shape").elem_cnt();
    const auto& p2p_pair = kernel_state->p2p_pair();

    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0,
                        total_elem_cnt * GetSizeOfDataType(out->data_type()));
    std::unique_ptr<ep::primitive::Add> add_primitive =
        ep::primitive::NewPrimitive<ep::primitive::AddFactory>(ctx->device_type(), in->data_type());
    CHECK(add_primitive);
    for (const auto& pair : p2p_pair) {
      int64_t src = pair.first;
      int64_t dst = pair.second;

      if (GlobalProcessCtx::Rank() == src) {
        CHECK_JUST(
            Send<device_type>(in_ptr, total_elem_cnt, in->data_type(), dst, ctx->device_ctx()));
      }
      if (GlobalProcessCtx::Rank() == dst) {
        CHECK_JUST(Recv<device_type>(tmp_buffer_ptr, total_elem_cnt, out->data_type(), src,
                                     ctx->device_ctx()));
        add_primitive->Launch(ctx->stream_ctx(), tmp_buffer_ptr, out->dptr(), out->mut_dptr(),
                              total_elem_cnt);
      }
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EAGER_P_TO_B_KERNEL(device)                \
  REGISTER_USER_KERNEL("eager_p_to_b")                      \
      .SetCreateFn<EagerPToBKernel<device>>()               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)) \
      .SetInferTmpSizeFn(InferEagerPToBKernelTmpBufferSize);

REGISTER_EAGER_P_TO_B_KERNEL(DeviceType::kCPU)
#if defined(WITH_CUDA) && HAS_GPU_SEND_RECV
REGISTER_EAGER_P_TO_B_KERNEL(DeviceType::kGPU)
#endif

}  // namespace oneflow
