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

#if CUDA_VERSION >= 11070

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

static size_t InferTmpBufferSizeForFlashAttentionGradKernel(InferContext* ctx) {
  const auto& q_shape = ctx->InputTensorDesc("query", 0).shape();
  const int batch_size = q_shape.At(0);
  const int seqlen_q = q_shape.At(1);
  const int num_heads = q_shape.At(2);
  const int head_size = q_shape.At(3);
  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_size_rounded = round_multiple(head_size, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);

  size_t buffer_size = 0;
  buffer_size += GetCudaAlignedSize(batch_size * num_heads * seqlen_q_rounded
                                    * GetSizeOfDataType(DataType::kFloat));
  buffer_size += GetCudaAlignedSize(batch_size * seqlen_q_rounded * num_heads * head_size_rounded
                                    * GetSizeOfDataType(DataType::kFloat));
  return buffer_size;
}

class ScaledDotProductFlashAttentionGradKernel final : public user_op::OpKernel,
                                                       public user_op::CudaGraphSupport {
 public:
  ScaledDotProductFlashAttentionGradKernel() = default;
  ~ScaledDotProductFlashAttentionGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* grad_out = ctx->Tensor4ArgNameAndIndex("grad_out", 0);
    const Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const Tensor* softmax_lse = ctx->Tensor4ArgNameAndIndex("softmax_lse", 0);
    const Tensor* rng_state = ctx->Tensor4ArgNameAndIndex("rng_state", 0);
    const Tensor* alibi_slopes_ = nullptr;
    if (ctx->has_input("alibi_slopes_", 0)) {
      alibi_slopes_ = ctx->Tensor4ArgNameAndIndex("alibi_slopes_", 0);
    }

    const float p_dropout = ctx->Attr<float>("p_dropout");
    const float softmax_scale = ctx->Attr<float>("softmax_scale");
    bool is_causal = ctx->Attr<bool>("is_causal");
    int window_size_left = ctx->Attr<int32_t>("window_size_left");
    int window_size_right = ctx->Attr<int32_t>("window_size_right");

    Tensor* grad_q = ctx->Tensor4ArgNameAndIndex("grad_q", 0);
    Tensor* grad_k = ctx->Tensor4ArgNameAndIndex("grad_k", 0);
    Tensor* grad_v = ctx->Tensor4ArgNameAndIndex("grad_v", 0);
    Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    void* tmp_ptr = tmp->mut_dptr();

    auto* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
    auto dprops = cuda_device->properties();
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    bool is_dropout = p_dropout > 0.0f;

    if (is_causal) { window_size_right = 0; }

    const int arch = cuda_stream->cuda_arch() / 10;
    const bool is_supported_arch = (arch == 80 || arch == 86 || arch == 89 || arch == 90);
    CHECK(is_supported_arch);

    const DataType data_type = query->data_type();
    const bool is_supported_dtype =
        (data_type == DataType::kFloat16 || data_type == DataType::kBFloat16);
    CHECK(is_supported_dtype);
    CHECK_EQ(key->data_type(), data_type);
    CHECK_EQ(value->data_type(), data_type);
    CHECK_EQ(grad_out->data_type(), data_type);
    CHECK_EQ(out->data_type(), data_type);
    CHECK_EQ(softmax_lse->data_type(), DataType::kFloat);
    CHECK_EQ(rng_state->data_type(), DataType::kUInt64);

    // check contiguous last dimension.
    CHECK_EQ(CHECK_JUST(VectorAt(grad_out->stride(), 3)), 1);
    CHECK_EQ(CHECK_JUST(VectorAt(query->stride(), 3)), 1);
    CHECK_EQ(CHECK_JUST(VectorAt(key->stride(), 3)), 1);
    CHECK_EQ(CHECK_JUST(VectorAt(value->stride(), 3)), 1);
    CHECK_EQ(CHECK_JUST(VectorAt(out->stride(), 3)), 1);

    const int batch_size = query->shape_view().At(0);
    const int seqlen_q = query->shape_view().At(1);
    const int num_heads = query->shape_view().At(2);
    const int head_size = query->shape_view().At(3);
    const int seqlen_k = key->shape_view().At(1);
    const int num_heads_k = key->shape_view().At(2);
    const int head_size_og = grad_out->shape_view().At(3);

    // check tensor shape.
    CHECK_EQ(grad_out->shape_view().At(0), batch_size);
    CHECK_EQ(grad_out->shape_view().At(1), seqlen_q);
    CHECK_EQ(grad_out->shape_view().At(2), num_heads);
    CHECK_EQ(grad_out->shape_view().At(3), head_size_og);
    CHECK_EQ(query->shape_view().At(0), batch_size);
    CHECK_EQ(query->shape_view().At(1), seqlen_q);
    CHECK_EQ(query->shape_view().At(2), num_heads);
    CHECK_EQ(query->shape_view().At(3), head_size);
    CHECK_EQ(key->shape_view().At(0), batch_size);
    CHECK_EQ(key->shape_view().At(1), seqlen_k);
    CHECK_EQ(key->shape_view().At(2), num_heads_k);
    CHECK_EQ(key->shape_view().At(3), head_size);
    CHECK_EQ(value->shape_view().At(0), batch_size);
    CHECK_EQ(value->shape_view().At(1), seqlen_k);
    CHECK_EQ(value->shape_view().At(2), num_heads_k);
    CHECK_EQ(value->shape_view().At(3), head_size);
    CHECK_EQ(out->shape_view().At(0), batch_size);
    CHECK_EQ(out->shape_view().At(1), seqlen_q);
    CHECK_EQ(out->shape_view().At(2), num_heads);
    CHECK_EQ(out->shape_view().At(3), head_size);
    CHECK_EQ(softmax_lse->shape_view().At(0), batch_size);
    CHECK_EQ(softmax_lse->shape_view().At(1), num_heads);
    CHECK_EQ(softmax_lse->shape_view().At(2), seqlen_q);

    CHECK_GT(batch_size, 0);   // batch size must be postive
    CHECK_LE(head_size, 256);  // only support head dimensions at most 256
    // FlashAttention backward for head dim 256 with dropout, or head dim 224 with/without dropout
    // requires A100/A800 or H100/H800
    if (head_size > 192 && (head_size <= 224 || is_dropout)) { CHECK((arch == 80 || arch == 90)); }
    CHECK(num_heads % num_heads_k
          == 0);  // Number of heads in key/value must devide number of heads in query

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    // size: batch_size x num_heads x seqlen_q_rounded; datatype: float
    void* softmax_d_ptr = tmp_ptr;
    tmp_ptr = reinterpret_cast<char*>(tmp_ptr)
              + GetCudaAlignedSize(batch_size * num_heads * seqlen_q_rounded
                                   * GetSizeOfDataType(DataType::kFloat));

    // set to false by default.
    // TODO(chende): can get from forward kernel(add input in python interface, it's only used for
    // backward).
    bool deterministic = false;

    void* dq_accum_ptr;
    if (loop) {
      // size: batch_size x seqlen_q_rounded x num_heads x head_size_rounded; datatype: float
      dq_accum_ptr = tmp_ptr;
    }

    Flash_bwd_params params;

    set_params_dgrad(params, batch_size, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k, head_size, head_size_rounded, query, key, value, out,
                     grad_out, grad_q, grad_k, grad_v, nullptr, nullptr,
                     loop ? dq_accum_ptr : nullptr,
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr, nullptr, const_cast<void*>(softmax_lse->dptr()), softmax_d_ptr,
                     p_dropout, softmax_scale, window_size_left, window_size_right, deterministic);

    params.dq_accum_split_stride =
        !deterministic ? 0 : seqlen_q_rounded * num_heads * head_size_rounded;

    auto launch = &run_mha_bwd;

    params.rng_state = const_cast<uint64_t*>(rng_state->dptr<uint64_t>());

    set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    if (seqlen_q > 0) { launch(params, cuda_stream->cuda_stream()); }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SCALED_DOT_PRODUCT_FLASH_ATTENTION_KERNEL(dtype)      \
  REGISTER_USER_KERNEL("scaled_dot_product_flash_attention_grad")      \
      .SetCreateFn<ScaledDotProductFlashAttentionGradKernel>()         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == dtype))   \
      .SetInferTmpSizeFn(InferTmpBufferSizeForFlashAttentionGradKernel);

REGISTER_SCALED_DOT_PRODUCT_FLASH_ATTENTION_KERNEL(DataType::kFloat16)
REGISTER_SCALED_DOT_PRODUCT_FLASH_ATTENTION_KERNEL(DataType::kBFloat16)

}  // namespace

}  // namespace user_op

}  // namespace oneflow

#endif  // WITH_CUTLASS

#endif  // CUDA_VERSION >= 11070
