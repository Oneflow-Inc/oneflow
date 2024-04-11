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
#include "oneflow/core/common/shape_view.h"
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
#include "flash.h"
#include "static_switch.h"

namespace oneflow {

namespace user_op {

namespace {

void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t b, const size_t seqlen_q, const size_t seqlen_k,
                      const size_t seqlen_q_rounded, const size_t seqlen_k_rounded, const size_t h,
                      const size_t h_k, const size_t d, const size_t d_rounded,
                      // device pointers
                      const Tensor* q, const Tensor* k, const Tensor* v, Tensor* out,
                      void* cu_seqlens_q_d, void* cu_seqlens_k_d, void* seqused_k, void* p_d,
                      void* softmax_lse_d, float p_dropout, float softmax_scale,
                      int window_size_left, int window_size_right,
                      bool seqlenq_ngroups_swapped = false) {
  // Reset the parameters
  std::memset(&params, 0, sizeof(params));

  params.is_bf16 = q->data_type() == DataType::kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = const_cast<void*>(q->dptr());
  params.k_ptr = const_cast<void*>(k->dptr());
  params.v_ptr = const_cast<void*>(v->dptr());
  // All stride are in elements, not bytes.
  params.q_row_stride = CHECK_JUST(VectorAt(q->stride(), 1));
  params.k_row_stride = CHECK_JUST(VectorAt(k->stride(), 1));
  params.v_row_stride = CHECK_JUST(VectorAt(v->stride(), 1));
  params.q_head_stride = CHECK_JUST(VectorAt(q->stride(), 2));
  params.k_head_stride = CHECK_JUST(VectorAt(k->stride(), 2));
  params.v_head_stride = CHECK_JUST(VectorAt(v->stride(), 2));
  params.o_ptr = out->mut_dptr();
  params.o_row_stride = CHECK_JUST(VectorAt(out->stride(), 1));
  params.o_head_stride = CHECK_JUST(VectorAt(out->stride(), 2));

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = CHECK_JUST(VectorAt(q->stride(), 0));
    params.k_batch_stride = CHECK_JUST(VectorAt(k->stride(), 0));
    params.v_batch_stride = CHECK_JUST(VectorAt(v->stride(), 0));
    params.o_batch_stride = CHECK_JUST(VectorAt(out->stride(), 0));
    if (seqlenq_ngroups_swapped) {
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int*>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  CHECK_LT(p_dropout, 1.f);
#ifdef FLASHATTENTION_DISABLE_DROPOUT
  TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
#endif

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
  if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

#ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
              "This flash attention build does not support local attention.");
#endif

  params.is_seqlens_k_cumulative = true;

#ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  TORCH_CHECK(d == d_rounded,
              "This flash attention build does not support headdim not being a multiple of 32.");
#endif
}

// void set_params_dgrad(Flash_bwd_params &params,
//                       // sizes
//                       const size_t b,
//                       const size_t seqlen_q,
//                       const size_t seqlen_k,
//                       const size_t seqlen_q_rounded,
//                       const size_t seqlen_k_rounded,
//                       const size_t h,
//                       const size_t h_k,
//                       const size_t d,
//                       const size_t d_rounded,
//                       // device pointers
//                       const Tensor q,
//                       const Tensor k,
//                       const Tensor v,
//                       const Tensor out,
//                       const Tensor dout,
//                       Tensor dq,
//                       Tensor dk,
//                       Tensor dv,
//                       void *cu_seqlens_q_d,
//                       void *cu_seqlens_k_d,
//                       void *dq_accum_d,
//                       void *dk_accum_d,
//                       void *dv_accum_d,
//                       void *softmax_lse_d,
//                       void *dsoftmax_sum_d,
//                       float p_dropout,
//                       float softmax_scale,
//                       int window_size_left,
//                       int window_size_right,
//                       bool deterministic) {
//
//   set_params_fprop(params,
//                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d,
//                     d_rounded, q, k, v, out, cu_seqlens_q_d, cu_seqlens_k_d, nullptr, nullptr,
//                     softmax_lse_d,
//                     p_dropout,
//                     softmax_scale,
//                     window_size_left,
//                     window_size_right);
//
//   // Set the pointers and strides.
//   params.do_ptr = dout.data_ptr();
//   params.do_row_stride = dout.stride(-3);
//   params.do_head_stride = dout.stride(-2);
//   params.dq_ptr = dq.data_ptr();
//   params.dk_ptr = dk.data_ptr();
//   params.dv_ptr = dv.data_ptr();
//   params.dq_row_stride = dq.stride(-3);
//   params.dk_row_stride = dk.stride(-3);
//   params.dv_row_stride = dv.stride(-3);
//   params.dq_head_stride = dq.stride(-2);
//   params.dk_head_stride = dk.stride(-2);
//   params.dv_head_stride = dv.stride(-2);
//
//   if (cu_seqlens_q_d == nullptr) {
//     params.do_batch_stride = dout.stride(0);
//     params.dq_batch_stride = dq.stride(0);
//     params.dk_batch_stride = dk.stride(0);
//     params.dv_batch_stride = dv.stride(0);
//   }
//
//   params.dq_accum_ptr = dq_accum_d;
//   params.dk_accum_ptr = dk_accum_d;
//   params.dv_accum_ptr = dv_accum_d;
//
//   // Softmax sum
//   params.dsoftmax_sum = dsoftmax_sum_d;
//
//   params.deterministic = deterministic;
// }

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream, bool force_split_kernel = false) {
  FP16_SWITCH(!params.is_bf16, [&] {
    HEADDIM_SWITCH(params.d, [&] {
      if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
        run_mha_fwd_<elem_type, kHeadDim>(params, stream);
      } else {
        run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
      }
    });
  });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks,
                                int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
  // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
  // (i.e. it's 11 splits anyway).
  // So we check if the number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1
           || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) { max_efficiency = eff; }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) { continue; }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

void set_params_splitkv(Flash_fwd_params& params, const int batch_size, const int num_heads,
                        const int head_size, const int max_seqlen_k, const int max_seqlen_q,
                        const int head_size_rounded, const float p_dropout, const int num_splits,
                        cudaDeviceProp& dprops, void* tmp_ptr) {
  // This needs to match with run_mha_fwd_splitkv_dispatch
  const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
  const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
  // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
  // In any case we don't expect seqlen_q to be larger than 64 for inference.
  const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
  params.num_splits = num_splits;
  if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
    if (num_splits < 1) {
      params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks,
                                               dprops.multiProcessorCount, num_n_blocks, 128);
    }
    if (params.num_splits > 1) {
      size_t softmax_lse_accum_size =
          params.num_splits * batch_size * num_heads * max_seqlen_q * sizeof(float);
      params.softmax_lseaccum_ptr = tmp_ptr;
      params.oaccum_ptr =
          reinterpret_cast<char*>(tmp_ptr) + GetCudaAlignedSize(softmax_lse_accum_size);
    }
    CHECK_LE(params.num_splits, 128);
  }
}

void set_params_alibi(Flash_fwd_params& params, const Tensor* alibi_slopes_, int batch_size,
                      int num_heads) {
  params.alibi_slopes_ptr = nullptr;
}

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
    const Tensor* out_ = nullptr;
    if (ctx->has_input("out_", 0)) { out_ = ctx->Tensor4ArgNameAndIndex("out_", 0); }
    const Tensor* alibi_slopes_ = nullptr;
    if (ctx->has_input("alibi_slopes_", 0)) {
      out_ = ctx->Tensor4ArgNameAndIndex("alibi_slopes_", 0);
    }

    const float p_dropout = ctx->Attr<float>("p_dropout");
    const float softmax_scale = ctx->Attr<float>("softmax_scale");
    bool is_causal = ctx->Attr<bool>("is_causal");
    int window_size_left = ctx->Attr<int32_t>("window_size_left");
    int window_size_right = ctx->Attr<int32_t>("window_size_right");
    uint64_t seed = ctx->Attr<int64_t>("seed");

    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Tensor* softmax_lse = ctx->Tensor4ArgNameAndIndex("softmax_lse", 0);
    Tensor* rng_state= ctx->Tensor4ArgNameAndIndex("rng_state", 0);
    Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    void* tmp_ptr = tmp->mut_dptr();

    auto* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
    auto dprops = cuda_device->properties();
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    const int arch = cuda_stream->cuda_arch() / 10;
    const bool is_supported_arch = (arch == 80 || arch == 86 || arch == 89 || arch == 90);
    CHECK(is_supported_arch);

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