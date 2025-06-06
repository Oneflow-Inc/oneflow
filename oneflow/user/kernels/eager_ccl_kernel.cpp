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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/user/kernels/collective_communication/include/communication_context.h"
#include "oneflow/user/kernels/collective_communication/include/all_reduce.h"
#include "oneflow/user/kernels/collective_communication/include/reduce_scatter.h"
#include "oneflow/user/kernels/collective_communication/include/all_gather.h"
#include "oneflow/user/kernels/collective_communication/include/reduce.h"
#include "oneflow/user/kernels/collective_communication/include/broadcast.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

auto AllReduceCollectiveCommunicationExists() {
  return hob::make_custom("AllReduceCollectiveCommunicationExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            DeviceType device_type = ctx.device_type();
                            return ccl::IsCommunicationContextRegistered(device_type)
                                   && ccl::IsAllReduceRegistered(device_type);
                          });
}

auto ReduceScatterCollectiveCommunicationExists() {
  return hob::make_custom("ReduceScatterCollectiveCommunicationExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            DeviceType device_type = ctx.device_type();
                            return ccl::IsCommunicationContextRegistered(device_type)
                                   && ccl::IsReduceScatterRegistered(device_type);
                          });
}

auto AllGatherCollectiveCommunicationExists() {
  return hob::make_custom("AllGatherCollectiveCommunicationExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            DeviceType device_type = ctx.device_type();
                            return ccl::IsCommunicationContextRegistered(device_type)
                                   && ccl::IsAllGatherRegistered(device_type);
                          });
}

auto ReduceCollectiveCommunicationExists() {
  return hob::make_custom("ReduceCollectiveCommunicationExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            DeviceType device_type = ctx.device_type();
                            return ccl::IsCommunicationContextRegistered(device_type)
                                   && ccl::IsReduceRegistered(device_type);
                          });
}

auto BroadcastCollectiveCommunicationExists() {
  return hob::make_custom("BroadcastCollectiveCommunicationExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            DeviceType device_type = ctx.device_type();
                            return ccl::IsCommunicationContextRegistered(device_type)
                                   && ccl::IsBroadcastRegistered(device_type);
                          });
}

class EagerCclOpKernelCache final : public user_op::OpKernelCache {
 public:
  explicit EagerCclOpKernelCache(user_op::KernelCacheContext* ctx) { Init(ctx); }
  ~EagerCclOpKernelCache() override = default;

  const std::shared_ptr<ccl::CommunicationContext>& communication_ctx() const {
    return communication_ctx_;
  }

 private:
  void Init(user_op::KernelCacheContext* ctx) {
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    ParallelConf parallel_conf;
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    Symbol<ParallelDesc> parallel_desc = SymbolOf(ParallelDesc(parallel_conf));
    communication_ctx_ = ccl::NewCommunicationContext(parallel_desc->device_type(), parallel_desc);
  }

  std::shared_ptr<ccl::CommunicationContext> communication_ctx_;
};

void InitEagerCclOpKernelCache(user_op::KernelCacheContext* ctx,
                               std::shared_ptr<user_op::OpKernelCache>* cache_ptr) {
  // NOTE(jianhao): the cache only depends on parallel_conf, and the kernel is singleton
  // once parallel_conf is determined, so only init the cache at the first time.
  if (*cache_ptr == nullptr) { *cache_ptr = std::make_shared<EagerCclOpKernelCache>(ctx); }
}

}  // namespace

class EagerCclAllReduceKernel final : public user_op::OpKernel {
 public:
  EagerCclAllReduceKernel() = default;
  ~EagerCclAllReduceKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    InitEagerCclOpKernelCache(ctx, cache_ptr);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* kernel_cache = dynamic_cast<const EagerCclOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->shape_view(), out->shape_view()) << kOfBugIssueUploadPrompt;
    CHECK_EQ(in->data_type(), out->data_type()) << kOfBugIssueUploadPrompt;

    ccl::ReduceType reduce_type = ccl::kSum;
    if (in->data_type() == kBool) { reduce_type = ccl::kMax; }

    std::unique_ptr<ccl::AllReduce> all_reduce = ccl::NewCollectiveCommunication<ccl::AllReduce>(
        ctx->device_type(), in->data_type(), reduce_type);
    all_reduce->Launch(ctx->stream(), in->dptr(), out->mut_dptr(), out->shape_view().elem_cnt(),
                       kernel_cache->communication_ctx());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_ccl_all_reduce")
    .SetCreateFn<EagerCclAllReduceKernel>()
    .SetIsMatchedHob(AllReduceCollectiveCommunicationExists());

class EagerCclReduceScatterKernel final : public user_op::OpKernel {
 public:
  EagerCclReduceScatterKernel() = default;
  ~EagerCclReduceScatterKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    InitEagerCclOpKernelCache(ctx, cache_ptr);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* kernel_cache = dynamic_cast<const EagerCclOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr) << kOfBugIssueUploadPrompt;
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type()) << kOfBugIssueUploadPrompt;
    const auto& op_type = ctx->Attr<std::string>("op_type");
    CHECK_EQ(op_type, "sum") << kOfBugIssueUploadPrompt;
    ccl::ReduceType reduce_type = ccl::kSum;
    if (in->data_type() == kBool) { reduce_type = ccl::kMax; }
    std::unique_ptr<ccl::ReduceScatter> reduce_scatter =
        ccl::NewCollectiveCommunication<ccl::ReduceScatter>(ctx->device_type(), in->data_type(),
                                                            reduce_type);
    reduce_scatter->Launch(ctx->stream(), in->dptr(), out->mut_dptr(), out->shape_view().elem_cnt(),
                           kernel_cache->communication_ctx());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_ccl_reduce_scatter")
    .SetCreateFn<EagerCclReduceScatterKernel>()
    .SetIsMatchedHob(ReduceScatterCollectiveCommunicationExists());

class EagerCclAllGatherKernel final : public user_op::OpKernel {
 public:
  EagerCclAllGatherKernel() = default;
  ~EagerCclAllGatherKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    InitEagerCclOpKernelCache(ctx, cache_ptr);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* kernel_cache = dynamic_cast<const EagerCclOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr) << kOfBugIssueUploadPrompt;
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type()) << kOfBugIssueUploadPrompt;
    std::unique_ptr<ccl::AllGather> all_gather =
        ccl::NewCollectiveCommunication<ccl::AllGather>(ctx->device_type(), in->data_type());
    all_gather->Launch(ctx->stream(), in->dptr(), out->mut_dptr(), in->shape_view().elem_cnt(),
                       kernel_cache->communication_ctx());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_ccl_all_gather")
    .SetCreateFn<EagerCclAllGatherKernel>()
    .SetIsMatchedHob(AllGatherCollectiveCommunicationExists());

class EagerCclReduceKernel final : public user_op::OpKernel {
 public:
  EagerCclReduceKernel() = default;
  ~EagerCclReduceKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    InitEagerCclOpKernelCache(ctx, cache_ptr);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* kernel_cache = dynamic_cast<const EagerCclOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t root = ctx->Attr<int64_t>("root");
    void* out_ptr = out->mut_dptr();
    if (GlobalProcessCtx::Rank() == root) {
      CHECK_EQ(in->shape_view(), out->shape_view());
      CHECK_EQ(in->data_type(), out->data_type());
    }
    if (out_ptr != nullptr) {
      CHECK_EQ(in->shape_view(), out->shape_view());
      CHECK_EQ(in->data_type(), out->data_type());
    }

    ccl::ReduceType reduce_type = ccl::kSum;
    if (in->data_type() == kBool) { reduce_type = ccl::kMax; }

    std::unique_ptr<ccl::Reduce> reduce = ccl::NewCollectiveCommunication<ccl::Reduce>(
        ctx->device_type(), in->data_type(), reduce_type);
    reduce->Launch(ctx->stream(), in->dptr(), out_ptr, in->shape_view().elem_cnt(), root,
                   kernel_cache->communication_ctx());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_ccl_reduce")
    .SetCreateFn<EagerCclReduceKernel>()
    .SetIsMatchedHob(ReduceCollectiveCommunicationExists());

class EagerCclBroadcastKernel final : public user_op::OpKernel {
 public:
  EagerCclBroadcastKernel() = default;
  ~EagerCclBroadcastKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    InitEagerCclOpKernelCache(ctx, cache_ptr);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    size_t size = ctx->input_size("in");
    CHECK_EQ(size, ctx->output_size("out"));
    for (int i = 0; i < size; ++i) { ComputeForOneInput(ctx, cache, i); }
  }
  void ComputeForOneInput(user_op::KernelComputeContext* ctx, const user_op::OpKernelCache* cache,
                          int index) const {
    auto* kernel_cache = dynamic_cast<const EagerCclOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr);
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", index);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", index);
    int64_t root = ctx->Attr<int64_t>("root");
    const void* in_ptr = in->dptr();
    if (GlobalProcessCtx::Rank() == root) {
      CHECK_EQ(in->shape_view(), out->shape_view());
      CHECK_EQ(in->data_type(), out->data_type());
    }
    if (in_ptr != nullptr) {
      CHECK_EQ(in->shape_view(), out->shape_view());
      CHECK_EQ(in->data_type(), out->data_type());
    }

    std::unique_ptr<ccl::Broadcast> broadcast =
        ccl::NewCollectiveCommunication<ccl::Broadcast>(ctx->device_type(), out->data_type());
    broadcast->Launch(ctx->stream(), in_ptr, out->mut_dptr(), out->shape_view().elem_cnt(), root,
                      kernel_cache->communication_ctx());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_ccl_broadcast")
    .SetCreateFn<EagerCclBroadcastKernel>()
    .SetIsMatchedHob(BroadcastCollectiveCommunicationExists());

class EagerCclTouchKernel final : public user_op::OpKernel {
 public:
  EagerCclTouchKernel() = default;
  ~EagerCclTouchKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override{
      // Do nothing.
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

REGISTER_USER_KERNEL("eager_ccl_touch")
    .SetCreateFn<EagerCclTouchKernel>()
    .SetIsMatchedHob(!(user_op::HobDeviceType() == DeviceType::kInvalidDevice)
                     && !(user_op::HobDeviceType() == DeviceType::kMockDevice));

namespace {

class EagerCclS2SCpuOpKernelCache final : public user_op::OpKernelCache {
 public:
  explicit EagerCclS2SCpuOpKernelCache(user_op::KernelCacheContext* ctx) { Init(ctx); }
  ~EagerCclS2SCpuOpKernelCache() override = default;

  Symbol<ParallelDesc> parallel_desc() const { return parallel_desc_; }

 private:
  void Init(user_op::KernelCacheContext* ctx) {
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    ParallelConf parallel_conf;
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    parallel_desc_ = SymbolOf(ParallelDesc(parallel_conf));
  }

  Symbol<ParallelDesc> parallel_desc_;
};

size_t InferEagerCclS2SCpuKernelTmpBufferSize(user_op::InferContext* ctx) {
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

template<typename T>
class EagerCclS2SCPUKernel final : public user_op::OpKernel {
 public:
  EagerCclS2SCPUKernel() = default;
  ~EagerCclS2SCPUKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    // NOTE(jianhao): the cache only depends on parallel_conf, and the kernel is singleton
    // once parallel_conf is determined, so only init the cache at the first time.
    if (*cache_ptr == nullptr) { *cache_ptr = std::make_shared<EagerCclS2SCpuOpKernelCache>(ctx); }
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* kernel_cache = dynamic_cast<const EagerCclS2SCpuOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr);
    // NOTE(hanbinbin): Compute logic copy from _nccl_logical_s2s
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t dtype_size = GetSizeOfDataType(in->data_type());
    int64_t data_size = in->shape_view().elem_cnt() * dtype_size;
    // NOTE: in (transpose)-> pack_to_ptr (all2all)-> unpack_from_ptr (transpose)-> out
    const char* pack_to_ptr = in->dptr<char>();
    char* unpack_from_ptr = out->mut_dptr<char>();
    int64_t tmp_size = tmp_buffer->shape_view().elem_cnt();
    CHECK_EQ(tmp_size, data_size * 2);

    CHECK_EQ(in->data_type(), out->data_type());
    const int64_t num_ranks = kernel_cache->parallel_desc()->parallel_num();
    CHECK_EQ(in->shape_view().elem_cnt(), out->shape_view().elem_cnt())
        << in->shape_view().ToString() << " vs " << out->shape_view().ToString();
    const int64_t elem_cnt = in->shape_view().elem_cnt();
    const int64_t in_split_axis = ctx->Attr<int64_t>("in_split_axis");
    const int64_t out_split_axis = ctx->Attr<int64_t>("out_split_axis");

    DimVector logical_shape_dim_vec;
    in->shape_view().ToDimVector(&logical_shape_dim_vec);
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
      std::vector<int32_t> perm;
      perm.emplace_back(out_split_axis);
      FOR_RANGE(int64_t, i, 0, transpose_in_dim_vec.size()) {
        if (i != out_split_axis) { perm.emplace_back(i); }
      }
      auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
          ctx->stream()->device_type(), transpose_in_dim_vec.size());
      CHECK(transpose);
      transpose->Launch(ctx->stream(), in->data_type(), transpose_in_dim_vec.size(),
                        transpose_in_dim_vec.data(), in->dptr(), perm.data(),
                        tmp_buffer->mut_dptr());
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
      const auto& p2p_pairs = CHECK_JUST(GroupP2PPair(kernel_cache->parallel_desc()));
      for (const auto& pair : *p2p_pairs) {
        int64_t src = pair.first;
        int64_t dst = pair.second;

        if (GlobalProcessCtx::Rank() == src) {
          Symbol<ParallelDesc> parallel_desc = kernel_cache->parallel_desc();
          int64_t device_id = GlobalProcessCtx::LocalRank(dst);
          int64_t parallel_id =
              CHECK_JUST(parallel_desc->ParallelId4MachineDeviceId(dst, device_id));

          CHECK_JUST(Send(reinterpret_cast<const void*>(reinterpret_cast<const char*>(pack_to_ptr)
                                                        + parallel_id * chunk_size),
                          elem_per_chunk, in->data_type(), dst, DeviceType::kCPU, ctx->stream()));
        }
        if (GlobalProcessCtx::Rank() == dst) {
          Symbol<ParallelDesc> parallel_desc = kernel_cache->parallel_desc();
          int64_t device_id = GlobalProcessCtx::LocalRank(src);
          int64_t parallel_id =
              CHECK_JUST(parallel_desc->ParallelId4MachineDeviceId(src, device_id));

          CHECK_JUST(Recv(reinterpret_cast<void*>(reinterpret_cast<char*>(unpack_from_ptr)
                                                  + parallel_id * chunk_size),
                          elem_per_chunk, out->data_type(), src, DeviceType::kCPU, ctx->stream()));
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
      std::vector<int32_t> perm;
      FOR_RANGE(int64_t, i, 1, unpack_from_dim_vec.size()) { perm.emplace_back(i); }
      perm.insert(perm.begin() + in_split_axis, 0);
      auto transpose = ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(
          ctx->stream()->device_type(), unpack_from_dim_vec.size());
      CHECK(transpose);
      transpose->Launch(ctx->stream(), in->data_type(), unpack_from_dim_vec.size(),
                        unpack_from_dim_vec.data(), unpack_from_ptr, perm.data(), out->mut_dptr());
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EAGER_CCL_S2S_CPU_KERNEL(dtype)                                         \
  REGISTER_USER_KERNEL("eager_ccl_s2s")                                                  \
      .SetCreateFn<EagerCclS2SCPUKernel<dtype>>()                                        \
      .SetIsMatchedHob(!(user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && HobIsSendAndRecvRegistered()                                   \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)   \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferEagerCclS2SCpuKernelTmpBufferSize);

REGISTER_EAGER_CCL_S2S_CPU_KERNEL(int8_t)
REGISTER_EAGER_CCL_S2S_CPU_KERNEL(int32_t)
REGISTER_EAGER_CCL_S2S_CPU_KERNEL(int64_t)
REGISTER_EAGER_CCL_S2S_CPU_KERNEL(bool)
REGISTER_EAGER_CCL_S2S_CPU_KERNEL(float)
REGISTER_EAGER_CCL_S2S_CPU_KERNEL(double)

#undef REGISTER_EAGER_CCL_S2S_KERNEL

}  // namespace oneflow
