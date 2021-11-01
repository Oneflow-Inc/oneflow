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
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/primitive/include/add.h"

namespace oneflow {

namespace {

Maybe<Symbol<cfg::NdSbp>> GetAllSplitNdSbp(int64_t axis, int64_t ndim) {
  cfg::NdSbp split_nd_sbp;
  for (int64_t i = 0; i < ndim; ++i) {
    split_nd_sbp.mutable_sbp_parallel()->Add()->mutable_split_parallel()->set_axis(axis);
  }
  return SymbolOf(split_nd_sbp);
}

auto* CachedGetAllSplitNdSbp = DECORATE(&GetAllSplitNdSbp, ThreadLocal);

Maybe<Symbol<cfg::NdSbp>> GetAllPartialSumNdSbp(int64_t ndim) {
  cfg::NdSbp split_nd_sbp;
  for (int64_t i = 0; i < ndim; ++i) {
    split_nd_sbp.mutable_sbp_parallel()->Add()->mutable_partial_sum_parallel();
  }
  return SymbolOf(split_nd_sbp);
}

auto* CachedGetAllPartialSumNdSbp = DECORATE(&GetAllPartialSumNdSbp, ThreadLocal);

class EagerPToSOpKernelState final : public user_op::OpKernelState {
 public:
  explicit EagerPToSOpKernelState(user_op::KernelInitContext* ctx) : elem_cnt_per_chunk_(0) {
    Init(ctx);
  }
  ~EagerPToSOpKernelState() override = default;

  int64_t elem_cnt_per_chunk() const { return elem_cnt_per_chunk_; }

  const std::vector<std::shared_ptr<TensorSliceCopier>>& sorted_in_tensor_slice_copier() const {
    return sorted_in_tensor_slice_copier_;
  }

  const std::vector<std::pair<int64_t, int64_t>>& sorted_p2p_pair() const {
    return sorted_p2p_pair_;
  }

 private:
  void Init(user_op::KernelInitContext* ctx) {
    const std::string& in_parallel_conf_txt = ctx->Attr<std::string>("in_parallel_conf");
    const std::string& out_parallel_conf_txt = ctx->Attr<std::string>("out_parallel_conf");
    const int64_t out_split_axis = ctx->Attr<int64_t>("out_split_axis");
    const Shape& shape = ctx->Attr<Shape>("shape");
    DeviceType device_type = ctx->device_type();
    DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
    Symbol<ParallelDesc> in_parallel_desc = CHECK_JUST(TxtStringToPlacement(in_parallel_conf_txt));
    Symbol<ParallelDesc> out_parallel_desc =
        CHECK_JUST(TxtStringToPlacement(out_parallel_conf_txt));
    int64_t out_parallel_num = out_parallel_desc->parallel_num();
    int64_t in_parallel_num = in_parallel_desc->parallel_num();
    elem_cnt_per_chunk_ = shape.elem_cnt() / out_parallel_num;
    for (int64_t out_parallel_id = 0; out_parallel_id < out_parallel_num; ++out_parallel_id) {
      int64_t dst = CHECK_JUST(out_parallel_desc->MachineId4ParallelId(out_parallel_id));
      const TensorSliceView& out_slice = GetTensorSliceView4ParallelId(
          *out_parallel_desc->hierarchy(),
          *CHECK_JUST(
              CachedGetAllSplitNdSbp(out_split_axis, out_parallel_desc->hierarchy()->NumAxes())),
          shape, out_parallel_id);
      CHECK(!out_slice.IsEmpty());
      for (int64_t in_parallel_id = 0; in_parallel_id < in_parallel_num; ++in_parallel_id) {
        int64_t src = CHECK_JUST(in_parallel_desc->MachineId4ParallelId(in_parallel_id));
        const TensorSliceView& in_slice = GetTensorSliceView4ParallelId(
            *in_parallel_desc->hierarchy(),
            *CHECK_JUST(CachedGetAllPartialSumNdSbp(in_parallel_desc->hierarchy()->NumAxes())),
            shape, in_parallel_id);
        CHECK(!in_slice.IsEmpty());
        const TensorSliceView& intersection = out_slice.Intersect(in_slice);
        CHECK(!intersection.IsEmpty());
        sorted_p2p_pair_.emplace_back(std::make_pair(src, dst));
        sorted_in_tensor_slice_copier_.emplace_back(
            std::make_shared<TensorSliceCopier>(intersection, in_slice, data_type, device_type));
      }
    }
  }

  int64_t elem_cnt_per_chunk_;
  std::vector<std::shared_ptr<TensorSliceCopier>> sorted_in_tensor_slice_copier_;
  std::vector<std::pair<int64_t, int64_t>> sorted_p2p_pair_;
};

size_t InferEagerPToSKernelTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  const Shape& shape = ctx->Attr<Shape>("shape");
  const std::string& out_parallel_conf_txt = ctx->Attr<std::string>("out_parallel_conf");
  Symbol<ParallelDesc> out_parallel_desc = CHECK_JUST(TxtStringToPlacement(out_parallel_conf_txt));
  int64_t out_parallel_num = out_parallel_desc->parallel_num();
  size_t tensor_byte_size =
      shape.elem_cnt() / out_parallel_num * GetSizeOfDataType(in_tensor.data_type());
  return tensor_byte_size;
}

}  // namespace

template<typename T, DeviceType device_type>
class EagerPToSKernel final : public user_op::OpKernel {
 public:
  EagerPToSKernel() = default;
  ~EagerPToSKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerPToSOpKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerPToSOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const void* in_ptr = in->dptr();
    void* tmp_buffer_ptr = tmp_buffer->mut_dptr();

    int64_t elem_cnt_per_chunk = kernel_state->elem_cnt_per_chunk();
    const auto& sorted_in_tensor_slice_copier = kernel_state->sorted_in_tensor_slice_copier();
    const auto& sorted_p2p_pair = kernel_state->sorted_p2p_pair();
    CHECK_EQ(sorted_in_tensor_slice_copier.size(), sorted_p2p_pair.size());

    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0,
                        elem_cnt_per_chunk * GetSizeOfDataType(out->data_type()));
    std::unique_ptr<primitive::Add> add_primitive =
        primitive::NewPrimitive<primitive::AddFactory>(ctx->device_type(), in->data_type());
    CHECK(add_primitive);
    for (int64_t i = 0; i < sorted_p2p_pair.size(); ++i) {
      const auto& p2p_pair = sorted_p2p_pair.at(i);
      int64_t src = p2p_pair.first;
      int64_t dst = p2p_pair.second;
      if (GlobalProcessCtx::Rank() == src) {
        const auto& tensor_slice_copier = sorted_in_tensor_slice_copier.at(i);
        tensor_slice_copier->Copy(ctx->stream_ctx(), tmp_buffer_ptr, in_ptr);
        CHECK_JUST(Send<device_type>(reinterpret_cast<const void*>(tmp_buffer_ptr),
                                     elem_cnt_per_chunk, in->data_type(), dst, ctx->device_ctx()));
      }
      if (GlobalProcessCtx::Rank() == dst) {
        CHECK_JUST(Recv<device_type>(tmp_buffer_ptr, elem_cnt_per_chunk, out->data_type(), src,
                                     ctx->device_ctx()));
        add_primitive->Launch(ctx->stream_ctx(), tmp_buffer_ptr, out->dptr(), out->mut_dptr(),
                              elem_cnt_per_chunk);
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EAGER_P_TO_S_KERNEL(dtype, device)                                     \
  REGISTER_USER_KERNEL("eager_p_to_s")                                                  \
      .SetCreateFn<EagerPToSKernel<dtype, device>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                              \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferEagerPToSKernelTmpBufferSize);

REGISTER_EAGER_P_TO_S_KERNEL(float, DeviceType::kCPU)
REGISTER_EAGER_P_TO_S_KERNEL(double, DeviceType::kCPU)
#if defined(WITH_CUDA) && HAS_GPU_SEND_RECV
REGISTER_EAGER_P_TO_S_KERNEL(float16, DeviceType::kGPU)
REGISTER_EAGER_P_TO_S_KERNEL(float, DeviceType::kGPU)
REGISTER_EAGER_P_TO_S_KERNEL(double, DeviceType::kGPU)
#endif

}  // namespace oneflow
