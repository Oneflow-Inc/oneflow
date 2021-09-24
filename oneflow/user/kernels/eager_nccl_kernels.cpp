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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

class EagerCclOpKernelState final : public user_op::OpKernelState {
 public:
  EagerCclOpKernelState(user_op::KernelInitContext* ctx) { Init(ctx); }
  ~EagerCclOpKernelState() override = default;

  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }

 private:
  void Init(user_op::KernelInitContext* ctx) {
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    ParallelConf parallel_conf;
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    parallel_desc_ = SymbolOf(ParallelDesc(parallel_conf));
  }

  Symbol<ParallelDesc> parallel_desc_;
};

size_t InferEagerCclS2SKernelTmpBufferSize(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->InputTensorDesc("in", 0);
  size_t tensor_byte_size = in_tensor.shape().elem_cnt() * GetSizeOfDataType(in_tensor.data_type());
  // NOTE(hanbinbin): Set tmp_buffer_size to twice tensor_byte_size because the
  // SbpParallel4ArgNameAndIndex function of LocalUserOpInferContext is unimplemented
  return tensor_byte_size * 2;
}

Maybe<std::vector<std::pair<int64_t, int64_t>>> RawGroupP2PPair(
    Symbol<ParallelDesc> parallel_desc) {
  std::shared_ptr<std::vector<std::pair<int64_t, int64_t>>> p2p_pairs =
      std::make_shared<std::vector<std::pair<int64_t, int64_t>>>();
  for (int64_t src : parallel_desc->sorted_machine_ids()) {
    for (int64_t dst : parallel_desc->sorted_machine_ids()) {
      p2p_pairs->emplace_back(std::make_pair(src, dst));
    }
  }
  return p2p_pairs;
}

static constexpr auto* GroupP2PPair = DECORATE(&RawGroupP2PPair, ThreadLocal);

}  // namespace

class EagerCclBroadcastKernel final : public user_op::OpKernel {
 public:
  EagerCclBroadcastKernel() = default;
  ~EagerCclBroadcastKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerCclOpKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerCclOpKernelState*>(state);
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
    CHECK_JUST(ccl::Broadcast<DeviceType::kCPU>(in_ptr, out->mut_dptr(), out->shape().elem_cnt(),
                                                out->data_type(), root,
                                                kernel_state->parallel_desc(), ctx->device_ctx()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_broadcast")
    .SetCreateFn<EagerCclBroadcastKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "cpu");

class EagerCclAllReduceKernel final : public user_op::OpKernel {
 public:
  EagerCclAllReduceKernel() = default;
  ~EagerCclAllReduceKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerCclOpKernelState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerCclOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape(), out->shape());
    CHECK_EQ(in->data_type(), out->data_type());

    CHECK_JUST(ccl::AllReduce<DeviceType::kCPU>(
        in->dptr(), out->mut_dptr(), out->shape().elem_cnt(), out->data_type(), ccl::kSum,
        kernel_state->parallel_desc(), ctx->device_ctx()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_all_reduce")
    .SetCreateFn<EagerCclAllReduceKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "cpu");

class EagerCclReduceScatterKernel final : public user_op::OpKernel {
 public:
  EagerCclReduceScatterKernel() = default;
  ~EagerCclReduceScatterKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerCclOpKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerCclOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    const auto& op_type = ctx->Attr<std::string>("op_type");
    CHECK_EQ(op_type, "sum");
    CHECK_JUST(ccl::ReduceScatter<DeviceType::kCPU>(
        in->dptr(), out->mut_dptr(), out->shape().elem_cnt(), out->data_type(), ccl::kSum,
        kernel_state->parallel_desc(), ctx->device_ctx()));
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_reduce_scatter")
    .SetCreateFn<EagerCclReduceScatterKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "cpu");

class EagerCclAllGatherKernel final : public user_op::OpKernel {
 public:
  EagerCclAllGatherKernel() = default;
  ~EagerCclAllGatherKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerCclOpKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerCclOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    CHECK_JUST(ccl::AllGather<DeviceType::kCPU>(in->dptr(), out->mut_dptr(), in->shape().elem_cnt(),
                                                out->data_type(), kernel_state->parallel_desc(),
                                                ctx->device_ctx()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_nccl_all_gather")
    .SetCreateFn<EagerCclAllGatherKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "cpu");

template<typename T>
class EagerCclS2SKernel final : public user_op::OpKernel {
 public:
  EagerCclS2SKernel() = default;
  ~EagerCclS2SKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<EagerCclOpKernelState>(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<EagerCclOpKernelState*>(state);
    CHECK(kernel_state != nullptr);
    // NOTE(hanbinbin): Compute logic copy from _nccl_logical_s2s
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = in->shape().elem_cnt() * dtype_size;
    // NOTE: in (transpose)-> pack_to_ptr (all2all)-> unpack_from_ptr (transpose)-> out
    const char* pack_to_ptr = in->dptr<char>();
    char* unpack_from_ptr = out->mut_dptr<char>();
    int64_t tmp_size = tmp_buffer->shape().elem_cnt();
    CHECK_EQ(tmp_size, data_size * 2);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = kernel_state->parallel_desc()->parallel_num();
    CHECK_EQ(in->shape().elem_cnt(), out->shape().elem_cnt())
        << in->shape().ToString() << " vs " << out->shape().ToString();
    const int64_t elem_cnt = in->shape().elem_cnt();
    const int64_t in_split_axis = ctx->Attr<int64_t>("in_split_axis");
    const int64_t out_split_axis = ctx->Attr<int64_t>("out_split_axis");

    DimVector logical_shape_dim_vec;
    in->shape().ToDimVector(&logical_shape_dim_vec);
    logical_shape_dim_vec[in_split_axis] = logical_shape_dim_vec.at(in_split_axis) * num_ranks;

    if (out_split_axis != 0) {
      // Do pack. Need transpose in -> pack_to
      // pack use temp buffer offset: [0, data_size]
      pack_to_ptr = tmp_buffer->dptr<char>();
      DimVector transpose_in_dim_vec = logical_shape_dim_vec;
      CHECK_EQ(transpose_in_dim_vec.at(in_split_axis) % num_ranks, 0);
      transpose_in_dim_vec[in_split_axis] = transpose_in_dim_vec.at(in_split_axis) / num_ranks;
      CHECK_EQ(transpose_in_dim_vec.at(out_split_axis) % num_ranks, 0);
      transpose_in_dim_vec[out_split_axis] = transpose_in_dim_vec.at(out_split_axis) / num_ranks;
      transpose_in_dim_vec.insert(transpose_in_dim_vec.begin() + out_split_axis, num_ranks);
      const Shape transpose_in_shape(transpose_in_dim_vec);
      DimVector pack_to_dim_vec;
      std::vector<int32_t> perm;
      perm.push_back(out_split_axis);
      pack_to_dim_vec.push_back(transpose_in_shape.At(out_split_axis));
      FOR_RANGE(int64_t, i, 0, transpose_in_shape.NumAxes()) {
        if (i != out_split_axis) {
          perm.push_back(i);
          pack_to_dim_vec.push_back(transpose_in_shape.At(i));
        }
      }
      CHECK_EQ(elem_cnt, transpose_in_shape.elem_cnt());
      const Shape pack_to_shape(pack_to_dim_vec);
      CHECK_EQ(elem_cnt, pack_to_shape.elem_cnt());
      NewKernelUtil<DeviceType::kCPU>::Transpose(
          ctx->device_ctx(), transpose_in_shape.NumAxes(), transpose_in_shape, pack_to_shape, perm,
          elem_cnt, in->dptr<T>(), reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>()));
    }

    if (in_split_axis != 0) {
      // Do unpack. Need transpose unpack_from -> out
      // unpack use temp buffer offset: [tmp_size - data_size, tmp_size]
      unpack_from_ptr = tmp_buffer->mut_dptr<char>() + (tmp_size - data_size);
    }

    {
      // NOTE: Do S2S
      const int64_t elem_per_chunk = elem_cnt / num_ranks;
      const int64_t chunk_size = elem_per_chunk * dtype_size;
      const auto& p2p_pairs = CHECK_JUST(GroupP2PPair(kernel_state->parallel_desc()));
      for (const auto& pair : *p2p_pairs) {
        int64_t src = pair.first;
        int64_t dst = pair.second;

        if (GlobalProcessCtx::Rank() == src) {
          Symbol<ParallelDesc> parallel_desc = kernel_state->parallel_desc();
          int64_t device_id = GlobalProcessCtx::LocalRank(dst);
          int64_t parallel_id =
              CHECK_JUST(parallel_desc->ParallelId4MachineDeviceId(dst, device_id));

          CHECK_JUST(Send<DeviceType::kCPU>(
              reinterpret_cast<const void*>(reinterpret_cast<const char*>(pack_to_ptr)
                                            + parallel_id * chunk_size),
              elem_per_chunk, in->data_type(), dst, ctx->device_ctx()));
        }
        if (GlobalProcessCtx::Rank() == dst) {
          Symbol<ParallelDesc> parallel_desc = kernel_state->parallel_desc();
          int64_t device_id = GlobalProcessCtx::LocalRank(src);
          int64_t parallel_id =
              CHECK_JUST(parallel_desc->ParallelId4MachineDeviceId(src, device_id));

          CHECK_JUST(Recv<DeviceType::kCPU>(
              reinterpret_cast<void*>(reinterpret_cast<char*>(unpack_from_ptr)
                                      + parallel_id * chunk_size),
              elem_per_chunk, out->data_type(), src, ctx->device_ctx()));
        }
      }
    }

    if (in_split_axis != 0) {
      // Do unpack.
      CHECK(unpack_from_ptr != out->mut_dptr<char>());
      DimVector unpack_from_dim_vec = logical_shape_dim_vec;
      CHECK_EQ(unpack_from_dim_vec.at(in_split_axis) % num_ranks, 0);
      unpack_from_dim_vec[in_split_axis] = unpack_from_dim_vec.at(in_split_axis) / num_ranks;
      CHECK_EQ(unpack_from_dim_vec.at(out_split_axis) % num_ranks, 0);
      unpack_from_dim_vec[out_split_axis] = unpack_from_dim_vec.at(out_split_axis) / num_ranks;
      unpack_from_dim_vec.insert(unpack_from_dim_vec.begin(), num_ranks);
      const Shape unpack_from_shape(unpack_from_dim_vec);
      DimVector transpose_out_dim_vec;
      std::vector<int32_t> perm;
      FOR_RANGE(int64_t, i, 1, unpack_from_shape.NumAxes()) {
        perm.push_back(i);
        transpose_out_dim_vec.push_back(unpack_from_shape.At(i));
      }
      perm.insert(perm.begin() + in_split_axis, 0);
      transpose_out_dim_vec.insert(transpose_out_dim_vec.begin() + in_split_axis,
                                   unpack_from_shape.At(0));
      const Shape transpose_out_shape(transpose_out_dim_vec);
      NewKernelUtil<DeviceType::kCPU>::Transpose(
          ctx->device_ctx(), unpack_from_shape.NumAxes(), unpack_from_shape, transpose_out_shape,
          perm, unpack_from_shape.elem_cnt(), reinterpret_cast<const T*>(unpack_from_ptr),
          out->mut_dptr<T>());
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EAGER_CCL_S2S_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("eager_nccl_s2s")                                                \
      .SetCreateFn<EagerCclS2SKernel<dtype>>()                                          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                               \
                       & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferEagerCclS2SKernelTmpBufferSize);

REGISTER_EAGER_CCL_S2S_KERNEL(int8_t)
REGISTER_EAGER_CCL_S2S_KERNEL(int32_t)
REGISTER_EAGER_CCL_S2S_KERNEL(int64_t)
REGISTER_EAGER_CCL_S2S_KERNEL(float)
REGISTER_EAGER_CCL_S2S_KERNEL(double)

}  // namespace oneflow
