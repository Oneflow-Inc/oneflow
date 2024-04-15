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

#include <cstddef>
#include <cstdint>
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/user_op_tensor.h"
#ifdef WITH_CUTLASS

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/mma.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/scaled_dot_product_attention_kernel.h"
// from flash_attention
#include "oneflow/user/kernels/scaled_dot_product_attention_util.h"

namespace oneflow {

namespace user_op {

namespace {

static size_t InferTmpBufferSizeForFlashAttentionKernel(InferContext* ctx) {
  const float p_dropout = ctx->Attr<float>("p_dropout");
  const auto& q_shape = ctx->InputTensorDesc("query", 0).shape();
  const auto& k_shape = ctx->InputTensorDesc("key", 0).shape();
  const int batch_size = q_shape.At(0);
  const int seqlen_q = q_shape.At(1);
  const int num_heads = q_shape.At(2);
  const int head_size_og = q_shape.At(3);
  const int seqlen_k = k_shape.At(1);
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size = round_multiple(head_size_og, 8);
  const int head_size_rounded = round_multiple(head_size, 32);

  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
  const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
  const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
  size_t buffer_size = 0;
  // for splitKV and splitKV is not implemented for dropout.
  if (p_dropout == 0.0f) {
    int num_splits =
        num_splits_heuristic(batch_size * num_heads * num_m_blocks, sm_count, num_n_blocks, 128);
    buffer_size += GetCudaAlignedSize(num_splits * batch_size * num_heads * seqlen_q
                                      * GetSizeOfDataType(DataType::kFloat));
    buffer_size += GetCudaAlignedSize(num_splits * batch_size * num_heads * seqlen_q
                                      * head_size_rounded * GetSizeOfDataType(DataType::kFloat));
  }
  return buffer_size;
}

class ScaledDotProductFlashAttentionKernel final : public user_op::OpKernel,
                                                   public user_op::CudaGraphSupport {
 public:
  ScaledDotProductFlashAttentionKernel() = default;
  ~ScaledDotProductFlashAttentionKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA));
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<ScaledDotProductFlashAttentionKernelState>(generator);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const Tensor* alibi_slopes_ = nullptr;
    if (ctx->has_input("alibi_slopes_", 0)) {
      // default to null, it will never get input for current flash-attn version.
      alibi_slopes_ = ctx->Tensor4ArgNameAndIndex("alibi_slopes_", 0);
      CHECK(!alibi_slopes_) << "alibi_slopes should not have value";
    }

    const float p_dropout = ctx->Attr<float>("p_dropout");
    const float softmax_scale = ctx->Attr<float>("softmax_scale");
    bool is_causal = ctx->Attr<bool>("is_causal");
    int window_size_left = ctx->Attr<int32_t>("window_size_left");
    int window_size_right = ctx->Attr<int32_t>("window_size_right");
    uint64_t seed = ctx->Attr<int64_t>("seed");

    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Tensor* softmax_lse = ctx->Tensor4ArgNameAndIndex("softmax_lse", 0);
    Tensor* rng_state = ctx->Tensor4ArgNameAndIndex("rng_state", 0);
    Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    void* tmp_ptr = tmp->mut_dptr();

    auto* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
    auto dprops = cuda_device->properties();
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    const int arch = cuda_stream->cuda_arch() / 10;
    const bool is_supported_arch = (arch == 80 || arch == 86 || arch == 89 || arch == 90);
    CHECK(is_supported_arch) << "only supports CUDA Arch 80, 86, 89 and 90.";

    const DataType data_type = query->data_type();
    const bool is_supported_dtype =
        (data_type == DataType::kFloat16 || data_type == DataType::kBFloat16);
    CHECK(is_supported_dtype);
    CHECK_EQ(key->data_type(), data_type);
    CHECK_EQ(value->data_type(), data_type);
    CHECK_EQ(out->data_type(), data_type);

    CHECK_EQ(softmax_lse->data_type(), DataType::kFloat);

    // check contiguous last dimension.
    CHECK_EQ(CHECK_JUST(VectorAt(query->stride(), 3)), 1);
    CHECK_EQ(CHECK_JUST(VectorAt(key->stride(), 3)), 1);
    CHECK_EQ(CHECK_JUST(VectorAt(value->stride(), 3)), 1);

    const int batch_size = query->shape_view().At(0);
    const int seqlen_q = query->shape_view().At(1);
    const int num_heads = query->shape_view().At(2);
    const int head_size_og = query->shape_view().At(3);
    const int seqlen_k = key->shape_view().At(1);
    const int num_heads_k = key->shape_view().At(2);

    // check tensor shape.
    CHECK_EQ(query->shape_view().At(0), batch_size);
    CHECK_EQ(query->shape_view().At(1), seqlen_q);
    CHECK_EQ(query->shape_view().At(2), num_heads);
    CHECK_EQ(query->shape_view().At(3), head_size_og);
    CHECK_EQ(key->shape_view().At(0), batch_size);
    CHECK_EQ(key->shape_view().At(1), seqlen_k);
    CHECK_EQ(key->shape_view().At(2), num_heads_k);
    CHECK_EQ(key->shape_view().At(3), head_size_og);
    CHECK_EQ(value->shape_view().At(0), batch_size);
    CHECK_EQ(value->shape_view().At(1), seqlen_k);
    CHECK_EQ(value->shape_view().At(2), num_heads_k);
    CHECK_EQ(value->shape_view().At(3), head_size_og);
    CHECK_EQ(out->shape_view().At(0), batch_size);
    CHECK_EQ(out->shape_view().At(1), seqlen_q);
    CHECK_EQ(out->shape_view().At(2), num_heads);
    CHECK_EQ(out->shape_view().At(3), head_size_og);
    CHECK_EQ(softmax_lse->shape_view().At(0), batch_size);
    CHECK_EQ(softmax_lse->shape_view().At(1), num_heads);
    CHECK_EQ(softmax_lse->shape_view().At(2), seqlen_q);

    CHECK_GT(batch_size, 0);      // batch size must be postive
    CHECK_LE(head_size_og, 256);  // only support head dimensions at most 256
    CHECK(num_heads % num_heads_k
          == 0);  // Number of heads in key/value must devide number of heads in query

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    const int seqlenq_ngroups_swapped = 0;

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    Flash_fwd_params params;
    set_params_fprop(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k, head_size, head_size_rounded, query, key, value, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*return_softmax=*/nullptr, softmax_lse->mut_dptr(), p_dropout, softmax_scale,
                     window_size_left, window_size_right);

    int64_t counter_offset = params.b * params.h * 32;
    params.rng_state = rng_state->mut_dptr<uint64_t>();

    set_params_splitkv(params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
                       head_size_rounded, p_dropout, /*num_splits*/ 0, dprops, tmp_ptr);

    if (p_dropout > 0.0f) {
      // todo gennerator.
      auto* flash_attention_kernel_state =
          dynamic_cast<ScaledDotProductFlashAttentionKernelState*>(state);
      CHECK_NOTNULL(flash_attention_kernel_state);
      const auto& generator = flash_attention_kernel_state->generator();
      CHECK_NOTNULL(generator);
      const auto device_index = cuda_device->device_index();
      std::shared_ptr<ep::CUDAGenerator> cuda_generator =
          CHECK_JUST(generator->Get<ep::CUDAGenerator>(device_index));
      params.philox_args =
          at::PhiloxCudaState(seed, cuda_generator->get_philox_offset(counter_offset));
    }

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_k > 0) { run_mha_fwd(params, cuda_stream->cuda_stream()); }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALED_DOT_PRODUCT_FLASH_ATTENTION_KERNEL(dtype)      \
  REGISTER_USER_KERNEL("scaled_dot_product_flash_attention")           \
      .SetCreateFn<ScaledDotProductFlashAttentionKernel>()             \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == dtype))   \
      .SetInferTmpSizeFn(InferTmpBufferSizeForFlashAttentionKernel);

REGISTER_SCALED_DOT_PRODUCT_FLASH_ATTENTION_KERNEL(DataType::kFloat16)
#if CUDA_VERSION >= 11000
REGISTER_SCALED_DOT_PRODUCT_FLASH_ATTENTION_KERNEL(DataType::kBFloat16)
#endif

}  // namespace

}  // namespace user_op

}  // namespace oneflow

#endif  // WITH_CUTLASS
