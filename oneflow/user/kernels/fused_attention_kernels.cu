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

#ifdef WITH_CUTLASS

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/warp/mma.h"
#include "kernel_forward.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "trt_flash_attention/fmha.h"
#include "trt_flash_attention/fmha_flash_attention.h"

namespace oneflow {

namespace user_op {

namespace {

void ParseDims(const ShapeView& shape, const std::string& layout,
               const Optional<int64_t>& batch_size, const Optional<int64_t>& seq_len,
               const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
               int64_t tensor_index, int64_t* b, int64_t* m, int64_t* h, int64_t* k,
               int64_t* b_stride, int64_t* m_stride, int64_t* h_stride, int64_t* offset,
               bool* bm_packed) {
  if (shape.NumAxes() == 2) {
    if (layout == "(BM)(HK)" || layout == "(BM)(H2K)" || layout == "(BM)(H3K)") {
      *bm_packed = true;
      CHECK(batch_size);
      CHECK(seq_len);
      *b = CHECK_JUST(batch_size);
      *m = CHECK_JUST(seq_len);
      int64_t packed_n = 0;
      if (layout == "(BM)(HK)") {
        packed_n = 1;
      } else if (layout == "(BM)(H2K)") {
        packed_n = 2;
      } else if (layout == "(BM)(H3K)") {
        packed_n = 3;
      } else {
        UNIMPLEMENTED();
      }
      const int64_t hidden_size = shape.At(1);
      if (num_heads) {
        const int64_t expected_h = CHECK_JUST(num_heads);
        const int64_t packed_h = packed_n * expected_h;
        CHECK_EQ(hidden_size % packed_h, 0);
        *h = expected_h;
        *k = hidden_size / packed_h;
      } else if (head_size) {
        const int64_t expected_k = CHECK_JUST(head_size);
        const int64_t packed_k = packed_n * expected_k;
        CHECK_EQ(hidden_size % packed_k, 0);
        *h = hidden_size / packed_k;
        *k = expected_k;
      } else {
        UNIMPLEMENTED();
      }
      *h_stride = *k * packed_n;
      *m_stride = *h_stride * *h;
      *b_stride = 0;
      if (packed_n == 1) {
        *offset = 0;
      } else if (packed_n == 2) {
        CHECK_GE(tensor_index, 1);
        *offset = (tensor_index - 1) * *k;
      } else if (packed_n == 3) {
        *offset = tensor_index * *k;
      } else {
        UNIMPLEMENTED();
      }
    } else {
      UNIMPLEMENTED();
    }
  } else if (shape.NumAxes() == 3) {
    if (layout == "BM(HK)" || layout == "BM(H2K)" || layout == "BM(H3K)" || layout == "MB(HK)"
        || layout == "MB(H2K)" || layout == "MB(H3K)") {
      *bm_packed = false;
      bool batch_first = false;
      int64_t packed_n = 0;
      const std::string layout_bm = layout.substr(0, 2);
      const std::string layout_hk = layout.substr(2);
      if (layout_bm == "BM") {
        *b = shape.At(0);
        *m = shape.At(1);
        batch_first = true;
      } else if (layout_bm == "MB") {
        *b = shape.At(1);
        *m = shape.At(0);
        batch_first = false;
      } else {
        UNIMPLEMENTED();
      }
      if (layout_hk == "(HK)") {
        packed_n = 1;
      } else if (layout_hk == "(H2K)") {
        packed_n = 2;
      } else if (layout_hk == "(H3K)") {
        packed_n = 3;
      } else {
        UNIMPLEMENTED();
      }
      const int64_t hidden_size = shape.At(2);
      if (num_heads) {
        const int64_t expected_h = CHECK_JUST(num_heads);
        const int64_t packed_h = packed_n * expected_h;
        CHECK_EQ(hidden_size % packed_h, 0);
        *h = expected_h;
        *k = hidden_size / packed_h;
      } else if (head_size) {
        const int64_t expected_k = CHECK_JUST(head_size);
        const int64_t packed_k = packed_n * expected_k;
        CHECK_EQ(hidden_size % packed_k, 0);
        *h = hidden_size / packed_k;
        *k = expected_k;
      } else {
        UNIMPLEMENTED();
      }
      *h_stride = *k * packed_n;
      if (batch_first) {
        *m_stride = *h_stride * *h;
        *b_stride = *m_stride * *m;
      } else {
        *b_stride = *h_stride * *h;
        *m_stride = *b_stride * *b;
      }
      if (packed_n == 1) {
        *offset = 0;
      } else if (packed_n == 2) {
        CHECK_GE(tensor_index, 1);
        *offset = (tensor_index - 1) * *k;
      } else if (packed_n == 3) {
        *offset = tensor_index * *k;
      } else {
        UNIMPLEMENTED();
      }
    } else if (layout == "(BM)HK") {
      *bm_packed = true;
      CHECK(batch_size);
      CHECK(seq_len);
      *b = CHECK_JUST(batch_size);
      *m = CHECK_JUST(seq_len);
      *h = shape.At(1);
      *k = shape.At(2);
      *h_stride = *k;
      *m_stride = *h_stride * *h;
      *b_stride = 0;
    } else {
      UNIMPLEMENTED();
    }
  } else if (shape.NumAxes() == 4) {
    *bm_packed = false;
    if (layout == "BMHK") {
      *b = shape.At(0);
      *m = shape.At(1);
      *h = shape.At(2);
      *k = shape.At(3);
      *h_stride = *k;
      *m_stride = *h_stride * *h;
      *b_stride = *m_stride * *m;
    } else if (layout == "BHMK") {
      *b = shape.At(0);
      *m = shape.At(2);
      *h = shape.At(1);
      *k = shape.At(3);
      *m_stride = *k;
      *h_stride = *m_stride * *m;
      *b_stride = *h_stride * *h;
    } else if (layout == "MBHK") {
      *b = shape.At(1);
      *m = shape.At(0);
      *h = shape.At(2);
      *k = shape.At(3);
      *h_stride = *k;
      *b_stride = *h_stride * *h;
      *m_stride = *b_stride * *b;
    } else {
      UNIMPLEMENTED();
    }
    *offset = 0;
  } else {
    UNIMPLEMENTED();
  };
  if (batch_size) {
    const int64_t expected_b = CHECK_JUST(batch_size);
    CHECK_EQ(*b, expected_b);
  }
  if (seq_len) {
    const int64_t expected_m = CHECK_JUST(seq_len);
    CHECK_EQ(*m, expected_m);
  }
  if (num_heads) {
    const int64_t expected_h = CHECK_JUST(num_heads);
    CHECK_EQ(*h, expected_h);
  }
  if (head_size) {
    const int64_t expected_k = CHECK_JUST(head_size);
    CHECK_EQ(*k, expected_k);
  }
}

void ParseDims(const ShapeView& shape, const std::string& layout,
               const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
               int64_t tensor_index, int64_t* b, int64_t* m, int64_t* h, int64_t* k,
               int64_t* b_stride, int64_t* m_stride, int64_t* h_stride, int64_t* offset) {
  bool bm_packed{};
  ParseDims(shape, layout, Optional<int64_t>(), Optional<int64_t>(), num_heads, head_size,
            tensor_index, b, m, h, k, b_stride, m_stride, h_stride, offset, &bm_packed);
}

template<typename T, int pack_size>
struct alignas(pack_size * sizeof(T)) Pack {
  T elem[pack_size];
};

template<typename T>
__global__ void PackQkv(int b, int s, int nh, int d, const T* q, const T* k, const T* v, T* o,
                        int32_t* seq_len) {
  int count = b * s * nh * d * 3;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += blockDim.x * gridDim.x) {
    int row = i / (d * 3);
    int out_col = i - row * (d * 3);
    T out;
    if (out_col < d) {
      out = q[row * d + out_col];
    } else if (out_col < 2 * d) {
      out = k[row * d + out_col - d];
    } else {
      out = v[row * d + out_col - d * 2];
    }
    o[i] = out;
  }
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < b + 1; i += blockDim.x * gridDim.x) {
    seq_len[i] = i * s;
  }
}

struct Params {
  DataType data_type;
  int64_t num_batches;
  int64_t num_heads;
  int64_t query_seq_len;
  int64_t kv_seq_len;
  int64_t head_size;
  int64_t value_head_size;
  int64_t q_stride_b;
  int64_t q_stride_m;
  int64_t q_stride_h;
  int64_t k_stride_b;
  int64_t k_stride_m;
  int64_t k_stride_h;
  int64_t v_stride_b;
  int64_t v_stride_m;
  int64_t v_stride_h;
  std::string attn_mask_type;
  int64_t causal_diagonal_offset;
  const void* query_ptr;
  const void* key_ptr;
  const void* value_ptr;
  const void* attn_bias_ptr;
  const void* query_seq_start_ptr;
  const void* key_seq_start_ptr;
  const void* key_seq_len_ptr;
  int64_t attn_bias_stride_b;
  int64_t attn_bias_stride_h;
  int64_t attn_bias_stride_m;
  void* out_ptr;
  void* workspace;
  int64_t workspace_size;
  float scale;
};

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block,
         bool single_value_iteration, bool with_attn_bias>
void LaunchCutlassFmha(const Params& params, ep::CudaStream* stream) {
  // The fmha implementation below is based on xformers's fmha
  // implementation at:
  // https://github.com/facebookresearch/xformers/tree/main/xformers/csrc/attention/cuda/fmha
  using Attention = AttentionKernel<T, ArchTag, is_aligned, queries_per_block, keys_per_block,
                                    single_value_iteration, false, with_attn_bias>;
  typename Attention::Params p{};
  p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query_ptr));
  p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key_ptr));
  p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value_ptr));
  p.attn_bias_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.attn_bias_ptr));
  p.seqstart_q_ptr =
      const_cast<int32_t*>(reinterpret_cast<const int32_t*>(params.query_seq_start_ptr));
  p.seqstart_k_ptr =
      const_cast<int32_t*>(reinterpret_cast<const int32_t*>(params.key_seq_start_ptr));
  p.seqlen_k_ptr = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(params.key_seq_len_ptr));
  p.logsumexp_ptr = nullptr;
  p.output_ptr = reinterpret_cast<T*>(params.out_ptr);
  if (Attention::kNeedsOutputAccumulatorBuffer) {
    using Acc = typename Attention::accum_t;
    CHECK_GE(params.workspace_size, params.num_batches * params.query_seq_len * params.num_heads
                                        * params.value_head_size * sizeof(Acc));
    p.output_accum_ptr = reinterpret_cast<Acc*>(params.workspace);
  } else {
    p.output_accum_ptr = nullptr;
  }
  p.num_heads = params.num_heads;
  p.num_batches = params.num_batches;
  p.head_dim = params.head_size;
  p.head_dim_value = params.value_head_size;
  p.num_queries = params.query_seq_len;
  p.num_keys = params.kv_seq_len;
  p.q_strideM = params.q_stride_m;
  p.k_strideM = params.k_stride_m;
  p.v_strideM = params.v_stride_m;
  p.o_strideM = p.head_dim_value * p.num_heads;
  p.bias_strideM = params.attn_bias_stride_m;

  p.q_strideH = params.q_stride_h;
  p.k_strideH = params.k_stride_h;
  p.v_strideH = params.v_stride_h;
  p.bias_strideH = params.attn_bias_stride_h;

  p.q_strideB = params.q_stride_b;
  p.k_strideB = params.k_stride_b;
  p.v_strideB = params.v_stride_b;
  p.bias_strideB = params.attn_bias_stride_b;

  p.scale = params.scale;

  if (params.attn_mask_type == "none") {
    p.custom_mask_type = Attention::NoCustomMask;
  } else if (params.attn_mask_type == "causal_from_top_left") {
    p.custom_mask_type = Attention::CausalFromTopLeft;
  } else if (params.attn_mask_type == "causal_from_bottom_right") {
    p.custom_mask_type = Attention::CausalFromBottomRight;
  } else {
    UNIMPLEMENTED();
  }
  p.causal_diagonal_offset = params.causal_diagonal_offset;
  p.use_dropout = false;

  constexpr auto kernel_fn = attention_kernel_batched_impl<Attention>;
  int smem_bytes = sizeof(typename Attention::SharedStorage);
  if (smem_bytes > 0xc000) {
    static bool once = [&]() {
      cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      return true;
    }();
  }
  CHECK(Attention::check_supported(p));
  kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream->cuda_stream()>>>(p);
}

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block,
         bool single_value_iteration>
void DispatchWithAttnBias(const Params& params, ep::CudaStream* stream) {
  if (params.attn_bias_ptr != nullptr) {
    LaunchCutlassFmha<T, ArchTag, is_aligned, queries_per_block, keys_per_block,
                      single_value_iteration, true>(params, stream);
  } else {
    LaunchCutlassFmha<T, ArchTag, is_aligned, queries_per_block, keys_per_block,
                      single_value_iteration, false>(params, stream);
  }
}

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block>
void DispatchSingleValueIteration(const Params& params, ep::CudaStream* stream) {
  if (params.value_head_size <= keys_per_block) {
    DispatchWithAttnBias<T, ArchTag, is_aligned, queries_per_block, keys_per_block, true>(params,
                                                                                          stream);
  } else {
    DispatchWithAttnBias<T, ArchTag, is_aligned, queries_per_block, keys_per_block, false>(params,
                                                                                           stream);
  }
}

template<typename T, typename ArchTag, bool is_aligned>
void DispatchKeysPerBlock(const Params& params, ep::CudaStream* stream) {
  if (params.value_head_size <= 64) {
    DispatchSingleValueIteration<T, ArchTag, is_aligned, 64, 64>(params, stream);
  } else {
    DispatchSingleValueIteration<T, ArchTag, is_aligned, 32, 128>(params, stream);
  }
}

template<typename T, typename ArchTag>
void DispatchIsAligned(const Params& params, ep::CudaStream* stream) {
  if (reinterpret_cast<uintptr_t>(params.query_ptr) % 16 == 0
      && reinterpret_cast<uintptr_t>(params.key_ptr) % 16 == 0
      && reinterpret_cast<uintptr_t>(params.value_ptr) % 16 == 0
      && params.attn_bias_stride_m % (16 / sizeof(T)) == 0
      && params.head_size % (16 / sizeof(T)) == 0
      && params.value_head_size % (16 / sizeof(T)) == 0) {
    DispatchKeysPerBlock<T, ArchTag, true>(params, stream);
  } else {
    DispatchKeysPerBlock<T, ArchTag, false>(params, stream);
  }
}

template<typename T>
void DispatchArchTag(const Params& params, ep::CudaStream* stream) {
  const int major = stream->device_properties().major;
  const int minor = stream->device_properties().minor;

  if (major == 8) {
    DispatchIsAligned<T, cutlass::arch::Sm80>(params, stream);
  } else if (major == 7) {
    if (minor == 5) {
      DispatchIsAligned<T, cutlass::arch::Sm75>(params, stream);
    } else {
      DispatchIsAligned<T, cutlass::arch::Sm70>(params, stream);
    }
  } else {
    UNIMPLEMENTED();
  }
}

void DispatchCutlassFmha(const Params& params, ep::CudaStream* stream) {
  if (params.data_type == DataType::kFloat16) {
    DispatchArchTag<cutlass::half_t>(params, stream);
  } else if (params.data_type == DataType::kFloat) {
    DispatchArchTag<float>(params, stream);
  } else {
    UNIMPLEMENTED();
  }
}

class FusedMultiHeadAttentionInferenceKernel final : public user_op::OpKernel,
                                                     public user_op::CudaGraphSupport {
 public:
  FusedMultiHeadAttentionInferenceKernel() = default;
  ~FusedMultiHeadAttentionInferenceKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const Tensor* attn_bias = nullptr;
    if (ctx->has_input("attn_bias", 0)) { attn_bias = ctx->Tensor4ArgNameAndIndex("attn_bias", 0); }
    const Tensor* query_seq_start = nullptr;
    const Tensor* key_seq_start = nullptr;
    const Tensor* key_seq_len = nullptr;
    const float scale = ctx->Attr<double>("scale");
    if (ctx->has_input("query_seq_start", 0)) {
      CHECK(ctx->has_input("key_seq_start", 0));
      query_seq_start = ctx->Tensor4ArgNameAndIndex("query_seq_start", 0);
      key_seq_start = ctx->Tensor4ArgNameAndIndex("key_seq_start", 0);
      CHECK(query_seq_start->data_type() == DataType::kInt32);
      CHECK(key_seq_start->data_type() == DataType::kInt32);
      CHECK_EQ(query_seq_start->shape_view().NumAxes(), 1);
      CHECK_GT(query_seq_start->shape_view().At(0), 1);
      CHECK(query_seq_start->shape_view() == key_seq_start->shape_view());
      if (ctx->has_input("key_seq_len", 0)) {
        key_seq_len = ctx->Tensor4ArgNameAndIndex("key_seq_len", 0);
        CHECK(key_seq_len->data_type() == DataType::kInt32);
        CHECK_EQ(key_seq_len->shape_view().NumAxes(), 1);
        CHECK_EQ(key_seq_len->shape_view().At(0), query_seq_start->shape_view().At(0) - 1);
      }
    } else {
      CHECK(!ctx->has_input("key_seq_start", 0));
      CHECK(!ctx->has_input("key_seq_len", 0));
    }
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const DataType data_type = query->data_type();
    CHECK_EQ(key->data_type(), data_type);
    CHECK_EQ(value->data_type(), data_type);
    CHECK_EQ(out->data_type(), data_type);
    const int64_t query_head_size = ctx->Attr<int64_t>("query_head_size");
    const std::string& attn_mask_type = ctx->Attr<std::string>("attn_mask_type");
    const int64_t causal_diagonal_offset = ctx->Attr<int64_t>("causal_diagonal_offset");
    CHECK_GE(causal_diagonal_offset, 0);
    const std::string& query_layout = ctx->Attr<std::string>("query_layout");
    const std::string& key_layout = ctx->Attr<std::string>("key_layout");
    const std::string& value_layout = ctx->Attr<std::string>("value_layout");
    const std::string& output_layout = ctx->Attr<std::string>("output_layout");

    Optional<int64_t> batch_size;
    if (query_seq_start != nullptr) { batch_size = query_seq_start->shape_view().At(0) - 1; }
    Optional<int64_t> query_max_seq_len;
    const int64_t attr_query_max_seq_len = ctx->Attr<int64_t>("query_max_seq_len");
    if (attr_query_max_seq_len != 0) { query_max_seq_len = attr_query_max_seq_len; }
    Optional<int64_t> key_max_seq_len;
    const int64_t attr_key_max_seq_len = ctx->Attr<int64_t>("key_max_seq_len");
    if (attr_key_max_seq_len != 0) { key_max_seq_len = attr_key_max_seq_len; }

    int64_t q_b = 0;
    int64_t q_m = 0;
    int64_t q_h = 0;
    int64_t q_k = 0;
    int64_t q_b_stride = 0;
    int64_t q_m_stride = 0;
    int64_t q_h_stride = 0;
    int64_t q_offset = 0;
    bool q_bm_packed = false;
    ParseDims(query->shape_view(), query_layout, batch_size, query_max_seq_len, Optional<int64_t>(),
              query_head_size, 0, &q_b, &q_m, &q_h, &q_k, &q_b_stride, &q_m_stride, &q_h_stride,
              &q_offset, &q_bm_packed);
    if (q_bm_packed) { CHECK(query_seq_start != nullptr); }

    int64_t k_b = 0;
    int64_t k_m = 0;
    int64_t k_h = 0;
    int64_t k_k = 0;
    int64_t k_b_stride = 0;
    int64_t k_m_stride = 0;
    int64_t k_h_stride = 0;
    int64_t k_offset = 0;
    bool k_bm_packed = false;
    ParseDims(key->shape_view(), key_layout, q_b, key_max_seq_len, Optional<int64_t>(),
              query_head_size, 1, &k_b, &k_m, &k_h, &k_k, &k_b_stride, &k_m_stride, &k_h_stride,
              &k_offset, &k_bm_packed);
    CHECK_EQ(k_b, q_b);
    CHECK_EQ(k_h, q_h);
    CHECK_EQ(k_bm_packed, q_bm_packed);

    int64_t v_b = 0;
    int64_t v_m = 0;
    int64_t v_h = 0;
    int64_t v_k = 0;
    int64_t v_b_stride = 0;
    int64_t v_m_stride = 0;
    int64_t v_h_stride = 0;
    int64_t v_offset = 0;
    bool v_bm_packed = false;
    ParseDims(value->shape_view(), value_layout, q_b, k_m, q_h, Optional<int64_t>(), 2, &v_b, &v_m,
              &v_h, &v_k, &v_b_stride, &v_m_stride, &v_h_stride, &v_offset, &v_bm_packed);
    CHECK_EQ(v_b, q_b);
    CHECK_EQ(v_m, k_m);
    CHECK_EQ(v_bm_packed, k_bm_packed);
    if (output_layout == "BM(HK)") {
      CHECK(!q_bm_packed);
      CHECK_EQ(out->shape_view().NumAxes(), 3);
      CHECK_EQ(out->shape_view().At(0), q_b);
      CHECK_EQ(out->shape_view().At(1), q_m);
      CHECK_EQ(out->shape_view().At(2), q_h * v_k);
    } else if (output_layout == "MB(HK)") {
      CHECK(!q_bm_packed);
      CHECK_EQ(out->shape_view().NumAxes(), 3);
      CHECK_EQ(q_b, 1);
      CHECK_EQ(out->shape_view().At(0), q_m);
      CHECK_EQ(out->shape_view().At(1), q_b);
      CHECK_EQ(out->shape_view().At(2), q_h * v_k);
    } else if (output_layout == "(BM)(HK)") {
      CHECK(q_bm_packed);
      CHECK_EQ(out->shape_view().NumAxes(), 2);
      CHECK_EQ(out->shape_view().At(0), query->shape_view().At(0));
      CHECK_EQ(out->shape_view().At(1), q_h * v_k);
    } else {
      UNIMPLEMENTED();
    }

    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    // Compatible with typo `KERENL`
    const bool enable_trt_flash_attn =
        ParseBooleanFromEnv(
            "ONEFLOW_KERNEL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL",
            ParseBooleanFromEnv("ONEFLOW_KERENL_FMHA_ENABLE_TRT_FLASH_ATTN_IMPL", true))
        && ParseBooleanFromEnv("ONEFLOW_MATMUL_ALLOW_HALF_PRECISION_ACCUMULATION", false);
    const bool is_default_scale =
        std::abs(scale - 1.0 / std::sqrt(static_cast<float>(q_k))) <= 1e-5;
    const int arch = cuda_stream->cuda_arch() / 10;
    const bool is_trt_supported_arch = (arch == 75 || arch == 80 || arch == 86 || arch == 89);
    const bool is_trt_supported_head_size = ((q_k == 40) || (q_k == 64));
    // Avoid PackQKV overhead when seq_len is small.
    const bool is_long_seq_len = q_m >= 512;
    const bool is_trt_supported_layout = (query_layout == "BMHK" || query_layout == "BM(HK)")
                                         && (key_layout == "BMHK" || key_layout == "BM(HK)")
                                         && (value_layout == "BMHK" || value_layout == "BM(HK)")
                                         && (output_layout == "BMHK" || output_layout == "BM(HK)");
    if (is_default_scale && query_seq_start == nullptr && enable_trt_flash_attn
        && data_type == DataType::kFloat16 && q_m == k_m && q_k == v_k && is_trt_supported_head_size
        && is_long_seq_len && is_trt_supported_arch && attn_mask_type == "none"
        && attn_bias == nullptr && is_trt_supported_layout) {
      // The fmha implementation below is based on TensorRT's multiHeadFlashAttentionPlugin
      // implementation at:
      // https://github.com/NVIDIA/TensorRT/tree/main/plugin/multiHeadFlashAttentionPlugin
      int32_t cu_seqlens_d_size = (q_b + 1) * sizeof(int32_t);
      int32_t* cu_seqlens_d = reinterpret_cast<int32_t*>(tmp->mut_dptr());
      half* packed_qkv =
          reinterpret_cast<half*>(tmp->mut_dptr<char>() + GetCudaAlignedSize(cu_seqlens_d_size));
      constexpr int pack_size = 4;
      using PackType = Pack<half, pack_size>;
      const int64_t count = q_b * q_m * q_h * q_k * 3 / pack_size;
      PackQkv<PackType><<<(count - 1 + 256) / 256, 256, 0, cuda_stream->cuda_stream()>>>(
          q_b, q_m, q_h, q_k / pack_size, reinterpret_cast<const PackType*>(query->dptr()),
          reinterpret_cast<const PackType*>(key->dptr()),
          reinterpret_cast<const PackType*>(value->dptr()), reinterpret_cast<PackType*>(packed_qkv),
          cu_seqlens_d);

#ifdef WITH_CUDA_GRAPHS
      cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
      if (cuda_stream->IsGraphCapturing()) {
        OF_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
      }
#endif  // WITH_CUDA_GRAPHS
      nvinfer1::plugin::FusedMultiHeadFlashAttentionKernel const* kernels =
          nvinfer1::plugin::getFMHAFlashCubinKernels(nvinfer1::plugin::DATA_TYPE_FP16, arch);
#ifdef WITH_CUDA_GRAPHS
      if (cuda_stream->IsGraphCapturing()) {
        OF_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
      }
#endif  // WITH_CUDA_GRAPHS
      nvinfer1::plugin::runFMHFAKernel(packed_qkv, cu_seqlens_d, out->mut_dptr(), q_b * q_m, arch,
                                       kernels, q_b, q_h, q_k, q_m, cuda_stream->cuda_stream());
      return;
    }

    Params params{};
    params.data_type = data_type;
    params.num_batches = q_b;
    params.num_heads = q_h;
    params.query_seq_len = q_m;
    params.kv_seq_len = k_m;
    params.head_size = q_k;
    params.value_head_size = v_k;
    params.scale = scale;
    params.q_stride_b = q_b_stride;
    params.q_stride_m = q_m_stride;
    params.q_stride_h = q_h_stride;
    params.k_stride_b = k_b_stride;
    params.k_stride_m = k_m_stride;
    params.k_stride_h = k_h_stride;
    params.v_stride_b = v_b_stride;
    params.v_stride_m = v_m_stride;
    params.v_stride_h = v_h_stride;
    params.query_ptr = query->dptr<char>() + q_offset * GetSizeOfDataType(data_type);
    params.key_ptr = key->dptr<char>() + k_offset * GetSizeOfDataType(data_type);
    params.value_ptr = value->dptr<char>() + v_offset * GetSizeOfDataType(data_type);
    params.query_seq_start_ptr =
        query_seq_start == nullptr ? nullptr : query_seq_start->dptr<int32_t>();
    params.key_seq_start_ptr = key_seq_start == nullptr ? nullptr : key_seq_start->dptr<int32_t>();
    params.key_seq_len_ptr = key_seq_len == nullptr ? nullptr : key_seq_len->dptr<int32_t>();
    params.out_ptr = out->mut_dptr();
    const int64_t tmp_buffer_size = tmp->shape_view().elem_cnt();
    params.workspace = tmp->mut_dptr();
    params.workspace_size = tmp_buffer_size;
    params.attn_mask_type = attn_mask_type;
    params.causal_diagonal_offset = causal_diagonal_offset;
    if (attn_bias != nullptr) {
      const int64_t num_attn_bias_axes = attn_bias->shape_view().NumAxes();
      CHECK_GE(num_attn_bias_axes, 1);
      CHECK_LE(num_attn_bias_axes, 4);
      DimVector padded_attn_bias_shape;
      for (int i = 0; i < 4 - num_attn_bias_axes; ++i) { padded_attn_bias_shape.push_back(1); }
      for (int i = 0; i < num_attn_bias_axes; ++i) {
        padded_attn_bias_shape.push_back(attn_bias->shape_view().At(i));
      }
      CHECK_GE(padded_attn_bias_shape.at(3), k_m);
      int64_t bias_stride = padded_attn_bias_shape.at(3);
      if (padded_attn_bias_shape.at(2) == 1) {
        params.attn_bias_stride_m = 0;
      } else {
        CHECK_GE(padded_attn_bias_shape.at(2), q_m);
        params.attn_bias_stride_m = bias_stride;
        bias_stride *= padded_attn_bias_shape.at(2);
      }
      if (padded_attn_bias_shape.at(1) == 1) {
        params.attn_bias_stride_h = 0;
      } else {
        CHECK_EQ(padded_attn_bias_shape.at(1), q_h);
        params.attn_bias_stride_h = bias_stride;
        bias_stride *= q_h;
      }
      if (padded_attn_bias_shape.at(0) == 1) {
        params.attn_bias_stride_b = 0;
      } else {
        CHECK_EQ(padded_attn_bias_shape.at(0), q_b);
        params.attn_bias_stride_b = bias_stride;
      }
      params.attn_bias_ptr = attn_bias->dptr();
    } else {
      params.attn_bias_ptr = nullptr;
      params.attn_bias_stride_m = 0;
      params.attn_bias_stride_h = 0;
      params.attn_bias_stride_b = 0;
    }
    DispatchCutlassFmha(params, cuda_stream);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t InferTmpBufferSize(InferContext* ctx) {
  const auto& out_desc = ctx->OutputTensorDesc("out", 0);
  size_t buffer_size = 0;
  buffer_size +=
      GetCudaAlignedSize(out_desc.shape().elem_cnt() * GetSizeOfDataType(DataType::kFloat));
  buffer_size +=
      GetCudaAlignedSize(out_desc.shape().elem_cnt() * GetSizeOfDataType(out_desc.data_type())) * 3;
  buffer_size +=
      GetCudaAlignedSize((out_desc.shape().At(0) + 1) * GetSizeOfDataType(DataType::kInt32));
  return buffer_size;
}

#define REGISTER_FUSED_MULTI_HEAD_ATTENTION_INFERENCE_KERNEL(dtype)    \
  REGISTER_USER_KERNEL("fused_multi_head_attention_inference")         \
      .SetCreateFn<FusedMultiHeadAttentionInferenceKernel>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == dtype))   \
      .SetInferTmpSizeFn(InferTmpBufferSize);

REGISTER_FUSED_MULTI_HEAD_ATTENTION_INFERENCE_KERNEL(DataType::kFloat16)
REGISTER_FUSED_MULTI_HEAD_ATTENTION_INFERENCE_KERNEL(DataType::kFloat)

template<typename Index>
struct ConcatParam {
  const void* past_ptr;
  const void* ptr;
  void* output_ptr;
  Index past_offset;
  Index offset;
  Index output_offset;
  Index past_m;
  Index past_stride_b;
  Index past_stride_m;
  Index past_stride_h;
  Index stride_b;
  Index stride_m;
  Index stride_h;
  Index output_stride_b;
  Index output_stride_m;
  Index output_stride_h;
  Index count;
  Index output_khm;
  Index output_kh;
  Index output_k;
};

template<typename Index>
struct BatchConcatParam {
  ConcatParam<Index> params[2];
};

template<typename T, typename Index>
__device__ void ConcatPastKeyValue(ConcatParam<Index> p) {
  for (Index i = blockIdx.x * blockDim.x + threadIdx.x; i < p.count; i += blockDim.x * gridDim.x) {
    Index b_idx = i / p.output_khm;
    Index b_off = i - b_idx * p.output_khm;
    Index m_idx = b_off / p.output_kh;
    Index m_off = b_off - m_idx * p.output_kh;
    Index h_idx = m_off / p.output_k;
    Index k_idx = m_off - h_idx * p.output_k;
    T v;
    if (m_idx < p.past_m) {
      v = reinterpret_cast<const T*>(
          p.past_ptr)[p.past_offset + b_idx * p.past_stride_b + m_idx * p.past_stride_m
                      + h_idx * p.past_stride_h + k_idx];
    } else {
      v = reinterpret_cast<const T*>(
          p.ptr)[p.offset + b_idx * p.stride_b + (m_idx - p.past_m) * p.stride_m
                 + h_idx * p.stride_h + k_idx];
    }
    reinterpret_cast<T*>(
        p.output_ptr)[p.output_offset + b_idx * p.output_stride_b + m_idx * p.output_stride_m
                      + h_idx * p.output_stride_h + k_idx] = v;
  }
}

template<size_t elem_size, typename Index>
__global__ void BatchConcatPastKeyValue(BatchConcatParam<Index> params) {
  if (blockIdx.y == 0) {
    ConcatPastKeyValue<std::aligned_storage<elem_size, elem_size>::type, Index>(params.params[0]);
  } else if (blockIdx.y == 1) {
    ConcatPastKeyValue<std::aligned_storage<elem_size, elem_size>::type, Index>(params.params[1]);
  } else {
    // do nothing
  }
}

class FusedAttentionConcatPastKeyValueKernel final : public user_op::OpKernel,
                                                     public user_op::CudaGraphSupport {
 public:
  FusedAttentionConcatPastKeyValueKernel() = default;
  ~FusedAttentionConcatPastKeyValueKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    Tensor* output_key = ctx->Tensor4ArgNameAndIndex("output_key", 0);
    Tensor* output_value = ctx->Tensor4ArgNameAndIndex("output_value", 0);
    const DataType data_type = key->data_type();
    const Tensor* past_key = nullptr;
    const Tensor* past_value = nullptr;
    if (ctx->has_input("past_key", 0)) {
      CHECK(ctx->has_input("past_value", 0));
      past_key = ctx->Tensor4ArgNameAndIndex("past_key", 0);
      past_value = ctx->Tensor4ArgNameAndIndex("past_value", 0);
      CHECK_EQ(past_key->data_type(), data_type);
      CHECK_EQ(past_value->data_type(), data_type);
    } else {
      CHECK(!ctx->has_input("past_value", 0));
    }
    CHECK_EQ(value->data_type(), data_type);
    CHECK_EQ(output_key->data_type(), data_type);
    CHECK_EQ(output_value->data_type(), data_type);
    const int64_t size_of_data_type = GetSizeOfDataType(data_type);
    const int64_t key_head_size = ctx->Attr<int64_t>("key_head_size");
    const std::string& past_key_layout = ctx->Attr<std::string>("past_key_layout");
    const std::string& past_value_layout = ctx->Attr<std::string>("past_value_layout");
    const std::string& key_layout = ctx->Attr<std::string>("key_layout");
    const std::string& value_layout = ctx->Attr<std::string>("value_layout");

    int64_t pack_size = 16 / size_of_data_type;
    while (key_head_size % pack_size != 0) { pack_size /= 2; }

    auto ParsePackedDims =
        [](const ShapeView& shape, const std::string& layout, const Optional<int64_t>& num_heads,
           const Optional<int64_t>& head_size, int64_t tensor_index, int64_t* b, int64_t* m,
           int64_t* h, int64_t* k, int64_t* b_stride, int64_t* m_stride, int64_t* h_stride,
           int64_t* offset, int64_t pack_size) {
          ParseDims(shape, layout, num_heads, head_size, tensor_index, b, m, h, k, b_stride,
                    m_stride, h_stride, offset);
          *k /= pack_size;
          *b_stride /= pack_size;
          *m_stride /= pack_size;
          *h_stride /= pack_size;
          *offset /= pack_size;
        };

    int64_t key_b = 0;
    int64_t key_m = 0;
    int64_t key_h = 0;
    int64_t key_k = 0;
    int64_t key_b_stride = 0;
    int64_t key_m_stride = 0;
    int64_t key_h_stride = 0;
    int64_t key_offset = 0;
    ParsePackedDims(key->shape_view(), key_layout, Optional<int64_t>(), key_head_size, 1, &key_b,
                    &key_m, &key_h, &key_k, &key_b_stride, &key_m_stride, &key_h_stride,
                    &key_offset, pack_size);

    int64_t value_b = 0;
    int64_t value_m = 0;
    int64_t value_h = 0;
    int64_t value_k = 0;
    int64_t value_b_stride = 0;
    int64_t value_m_stride = 0;
    int64_t value_h_stride = 0;
    int64_t value_offset = 0;
    ParsePackedDims(value->shape_view(), value_layout, key_h, key_head_size, 2, &value_b, &value_m,
                    &value_h, &value_k, &value_b_stride, &value_m_stride, &value_h_stride,
                    &value_offset, pack_size);
    CHECK_EQ(value_b, key_b);
    CHECK_EQ(value_m, key_m);

    int64_t past_key_b = 0;
    int64_t past_key_m = 0;
    int64_t past_key_h = 0;
    int64_t past_key_k = 0;
    int64_t past_key_b_stride = 0;
    int64_t past_key_m_stride = 0;
    int64_t past_key_h_stride = 0;
    int64_t past_key_offset = 0;
    if (past_key != nullptr) {
      ParsePackedDims(past_key->shape_view(), past_key_layout, key_h, key_head_size, 1, &past_key_b,
                      &past_key_m, &past_key_h, &past_key_k, &past_key_b_stride, &past_key_m_stride,
                      &past_key_h_stride, &past_key_offset, pack_size);
    }

    int64_t past_value_b = 0;
    int64_t past_value_m = 0;
    int64_t past_value_h = 0;
    int64_t past_value_k = 0;
    int64_t past_value_b_stride = 0;
    int64_t past_value_m_stride = 0;
    int64_t past_value_h_stride = 0;
    int64_t past_value_offset = 0;
    if (past_value != nullptr) {
      ParsePackedDims(past_value->shape_view(), past_value_layout, key_h, key_head_size, 2,
                      &past_value_b, &past_value_m, &past_value_h, &past_value_k,
                      &past_value_b_stride, &past_value_m_stride, &past_value_h_stride,
                      &past_value_offset, pack_size);
    }
    CHECK_EQ(past_value_b, past_key_b);
    CHECK_EQ(past_value_m, past_key_m);

    int64_t output_key_b = 0;
    int64_t output_key_m = 0;
    int64_t output_key_h = 0;
    int64_t output_key_k = 0;
    int64_t output_key_b_stride = 0;
    int64_t output_key_m_stride = 0;
    int64_t output_key_h_stride = 0;
    int64_t output_key_offset = 0;
    ParsePackedDims(output_key->shape_view(), past_key_layout, key_h, key_head_size, 1,
                    &output_key_b, &output_key_m, &output_key_h, &output_key_k,
                    &output_key_b_stride, &output_key_m_stride, &output_key_h_stride,
                    &output_key_offset, pack_size);
    CHECK_EQ(output_key_b, key_b);
    CHECK_EQ(output_key_m, past_key_m + key_m);

    int64_t output_value_b = 0;
    int64_t output_value_m = 0;
    int64_t output_value_h = 0;
    int64_t output_value_k = 0;
    int64_t output_value_b_stride = 0;
    int64_t output_value_m_stride = 0;
    int64_t output_value_h_stride = 0;
    int64_t output_value_offset = 0;
    ParsePackedDims(output_value->shape_view(), past_value_layout, key_h, key_head_size, 2,
                    &output_value_b, &output_value_m, &output_value_h, &output_value_k,
                    &output_value_b_stride, &output_value_m_stride, &output_value_h_stride,
                    &output_value_offset, pack_size);
    CHECK_EQ(output_value_b, key_b);
    CHECK_EQ(output_value_m, past_value_m + value_m);

    int64_t max_tensor_elem = (1 << 30) * pack_size;
    CHECK((past_key == nullptr || past_key->shape_view().elem_cnt() <= max_tensor_elem)
          && (past_value == nullptr || past_value->shape_view().elem_cnt() <= max_tensor_elem)
          && key->shape_view().elem_cnt() <= max_tensor_elem
          && value->shape_view().elem_cnt() <= max_tensor_elem
          && output_key->shape_view().elem_cnt() <= max_tensor_elem
          && output_value->shape_view().elem_cnt() <= max_tensor_elem);

    int64_t count = output_key_b * output_key_m * output_key_h * output_key_k;
    BatchConcatParam<int32_t> kv;

    kv.params[0].past_ptr = past_key == nullptr ? nullptr : past_key->dptr();
    kv.params[0].ptr = key->dptr();
    kv.params[0].output_ptr = output_key->mut_dptr();
    kv.params[0].past_offset = past_key_offset;
    kv.params[0].offset = key_offset;
    kv.params[0].output_offset = output_key_offset;
    kv.params[0].past_m = past_key_m;
    kv.params[0].past_stride_b = past_key_b_stride;
    kv.params[0].past_stride_m = past_key_m_stride;
    kv.params[0].past_stride_h = past_key_h_stride;
    kv.params[0].stride_b = key_b_stride;
    kv.params[0].stride_m = key_m_stride;
    kv.params[0].stride_h = key_h_stride;
    kv.params[0].output_stride_b = output_key_b_stride;
    kv.params[0].output_stride_m = output_key_m_stride;
    kv.params[0].output_stride_h = output_key_h_stride;
    kv.params[0].count = count;
    kv.params[0].output_khm = output_key_k * output_key_h * output_key_m;
    kv.params[0].output_kh = output_key_k * output_key_h;
    kv.params[0].output_k = output_key_k;

    kv.params[1].past_ptr = past_value == nullptr ? nullptr : past_value->dptr();
    kv.params[1].ptr = value->dptr();
    kv.params[1].output_ptr = output_value->mut_dptr();
    kv.params[1].past_offset = past_value_offset;
    kv.params[1].offset = value_offset;
    kv.params[1].output_offset = output_value_offset;
    kv.params[1].past_m = past_value_m;
    kv.params[1].past_stride_b = past_value_b_stride;
    kv.params[1].past_stride_m = past_value_m_stride;
    kv.params[1].past_stride_h = past_value_h_stride;
    kv.params[1].stride_b = value_b_stride;
    kv.params[1].stride_m = value_m_stride;
    kv.params[1].stride_h = value_h_stride;
    kv.params[1].output_stride_b = output_value_b_stride;
    kv.params[1].output_stride_m = output_value_m_stride;
    kv.params[1].output_stride_h = output_value_h_stride;
    kv.params[1].count = count;
    kv.params[1].output_khm = output_value_k * output_value_h * output_value_m;
    kv.params[1].output_kh = output_value_k * output_value_h;
    kv.params[1].output_k = output_value_k;

    constexpr uint32_t block_size = 256;
    const dim3 grid_size((count - 1 + block_size) / block_size, 2);

    const int64_t elem_size = size_of_data_type * pack_size;
    cudaStream_t cuda_stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    if (elem_size == 16) {
      BatchConcatPastKeyValue<16, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(kv);
    } else if (elem_size == 8) {
      BatchConcatPastKeyValue<8, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(kv);
    } else if (elem_size == 4) {
      BatchConcatPastKeyValue<4, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(kv);
    } else if (elem_size == 2) {
      BatchConcatPastKeyValue<2, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(kv);
    } else if (elem_size == 1) {
      BatchConcatPastKeyValue<1, int32_t><<<grid_size, block_size, 0, cuda_stream>>>(kv);
    } else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("fused_attention_concat_past_key_value")
    .SetCreateFn<FusedAttentionConcatPastKeyValueKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA));

template<typename T, typename PositionType, typename IndexType, size_t num_dims,
         size_t rotary_emb_dim>
struct FusedApplyRotaryEmbParam {
  const T* x;
  const T* cos;
  const T* sin;
  const PositionType* position_ids;
  T* out;
  const T theta;
  const float inv_actual_rotary_size;  // 1.0 / (rotary_size per rotary dimension)
  const IndexType actual_rotary_size;  // rotary_size per rotary dimension
  const IndexType rotary_size;
  const IndexType rotate_stride;
  const IndexType k0;
  const IndexType k1;
  IndexType num_elements;
  const IndexType k;
  const IndexType x_offset;

  IndexType ref_stride[num_dims];  // b, m, h, k
  IndexType out_stride[num_dims];  // ordered descendingly by stride
  IndexType x_stride[num_dims];

  IndexType position_b_stride;
  IndexType position_rotate_stride;

  IndexType sinuous_m_stride;

  FusedApplyRotaryEmbParam(const T* x, const T* cos, const T* sin, const PositionType* position_ids,
                           T* out, const T theta, const float inv_actual_rotary_size,
                           const IndexType actual_rotary_size, const IndexType rotary_size,
                           const IndexType rotate_stride, const IndexType num_elements,
                           const IndexType k, const IndexType k0, const IndexType k1,
                           const IndexType x_offset)
      : x(x),
        cos(cos),
        sin(sin),
        position_ids(position_ids),
        out(out),
        theta(theta),
        inv_actual_rotary_size(inv_actual_rotary_size),
        actual_rotary_size(actual_rotary_size),
        rotary_size(rotary_size),
        rotate_stride(rotate_stride),
        num_elements(num_elements),
        k(k),
        k0(k0),
        k1(k1),
        x_offset(x_offset) {}
};

template<typename T, typename PositionType, typename IndexType, size_t PackSize, size_t num_dims,
         size_t rotary_emb_dim>
__global__ void IntervalKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims, rotary_emb_dim> param) {
  for (IndexType packed_offset = threadIdx.x + blockIdx.x * blockDim.x;
       packed_offset < param.num_elements; packed_offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, PackSize>;
    IndexType offset = packed_offset * PackSize;
    IndexType index[num_dims];  // b, m, h, k

    IndexType temp_offset = offset;

    for (int i = 0; i < num_dims - 1; i++) {
      IndexType ref_stride = param.ref_stride[i];
      IndexType idx = temp_offset / ref_stride;
      index[i] = idx;
      temp_offset = temp_offset - idx * ref_stride;
    }
    index[num_dims - 1] = temp_offset;

    IndexType x_offset = param.x_offset;
    IndexType out_offset = 0;
#pragma unroll
    for (int i = 0; i < num_dims; i++) {
      x_offset = x_offset + param.x_stride[i] * index[i];
      out_offset = out_offset + param.out_stride[i] * index[i];
    }
    const LoadPack x_vec = *reinterpret_cast<const LoadPack*>(param.x + x_offset);

    const IndexType k_index = index[num_dims - 1];
    if (k_index < param.rotary_size) {
      const IndexType position_rotate_index = (k_index >= param.k0) ? 1 : 0;
      const IndexType b_index = index[0], m_index = index[1];
      const IndexType position_id_offset = b_index * param.position_b_stride
                                           + position_rotate_index * param.position_rotate_stride
                                           + m_index;

      const PositionType position =
          param.position_ids ? param.position_ids[position_id_offset] : m_index;
      const IndexType actual_k_index = k_index % param.actual_rotary_size;
      const IndexType sinuous_offset = position * param.sinuous_m_stride + actual_k_index;

      LoadPack cos_vec, sin_vec, out_vec;

      if (param.cos && param.sin) {
        cos_vec = *reinterpret_cast<const LoadPack*>(param.cos + sinuous_offset);
        sin_vec = *reinterpret_cast<const LoadPack*>(param.sin + sinuous_offset);
      } else {
        const IndexType actual_ndim = param.rotary_size / rotary_emb_dim;
#pragma unroll
        for (int i = 0; i < PackSize / 2; i++) {
          T val = position
                  * expf(2.0f * static_cast<float>(((actual_k_index >> 1) + i))
                         * param.inv_actual_rotary_size * logf(param.theta));
          T cos_val = cosf(val);
          T sin_val = sinf(val);
          cos_vec.elem[i * 2] = cos_val;
          cos_vec.elem[i * 2 + 1] = cos_val;
          sin_vec.elem[i * 2] = sin_val;
          sin_vec.elem[i * 2 + 1] = sin_val;
        }
      }

#pragma unroll
      for (int i = 0; i < PackSize / 2; i++) {
        out_vec.elem[i * 2] =
            x_vec.elem[i * 2] * cos_vec.elem[i * 2] - x_vec.elem[i * 2 + 1] * sin_vec.elem[i * 2];
        out_vec.elem[i * 2 + 1] = x_vec.elem[i * 2 + 1] * cos_vec.elem[i * 2 + 1]
                                  + x_vec.elem[i * 2] * sin_vec.elem[i * 2 + 1];
      }

      *(reinterpret_cast<LoadPack*>(param.out + out_offset)) = out_vec;
    } else {
      *(reinterpret_cast<LoadPack*>(param.out + out_offset)) = x_vec;
    }
  }
}

template<typename T, typename PositionType, typename IndexType, size_t num_dims,
         size_t rotary_emb_dim>
__global__ void PlaneKernel(
    FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims, rotary_emb_dim> param) {
  for (IndexType offset = threadIdx.x + blockIdx.x * blockDim.x; offset < param.num_elements;
       offset += blockDim.x * gridDim.x) {
    using LoadPack = cuda::elementwise::Packed<T, 2>;
    IndexType temp_offset = offset;
    IndexType index[num_dims];
#pragma unroll
    for (int i = 0; i < num_dims - 1; i++) {
      IndexType ref_stride = param.ref_stride[i];
      IndexType idx = temp_offset / ref_stride;
      index[i] = idx;
      temp_offset = temp_offset - idx * ref_stride;
    }
    index[num_dims - 1] = temp_offset;

    const IndexType b_index = index[0], m_index = index[1], k_index = index[num_dims - 1];
    const IndexType position_rotate_index = (k_index >= param.k0) ? 1 : 0;
    const IndexType position_id_offset = b_index * param.position_b_stride
                                         + position_rotate_index * param.position_rotate_stride
                                         + m_index;

    const PositionType position =
        param.position_ids ? param.position_ids[position_id_offset] : m_index;
    const IndexType actual_k_index = k_index % param.actual_rotary_size;
    const IndexType sinuous_offset = position * param.k + actual_k_index;

    T cos_val, sin_val, out_val;

    if (param.cos && param.sin) {
      cos_val = *(param.cos + sinuous_offset);
      sin_val = *(param.sin + sinuous_offset);
    } else {
      T val = position
              * expf(2.0f * static_cast<float>(k_index % (param.actual_rotary_size >> 1))
                     * param.inv_actual_rotary_size * logf(param.theta));
      cos_val = cosf(val);
      sin_val = sinf(val);
    }

    LoadPack x_vec;
    IndexType x_offset = param.x_offset;
    IndexType out_offset = 0;
#pragma unroll
    for (int i = 0; i < num_dims; i++) {
      x_offset = x_offset + param.x_stride[i] * index[i];
      out_offset = out_offset + param.out_stride[i] * index[i];
    }

    if (k_index < param.k0) {
      x_vec.elem[0] = *(param.x + x_offset);
      x_vec.elem[1] = (param.k0 - k_index > param.rotate_stride)
                          ? static_cast<T>(-*(param.x + x_offset + param.rotate_stride))
                          : *(param.x + x_offset - param.rotate_stride);
      out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
    } else if (k_index < param.k1) {
      x_vec.elem[0] = *(param.x + x_offset);
      x_vec.elem[1] = (param.k1 - k_index > param.rotate_stride)
                          ? static_cast<T>(-*(param.x + x_offset + param.rotate_stride))
                          : *(param.x + x_offset - param.rotate_stride);
      out_val = cos_val * x_vec.elem[0] + sin_val * x_vec.elem[1];
    } else {
      out_val = *(param.x + x_offset);
    }

    *(param.out + out_offset) = out_val;
  }
}

template<typename T, typename PositionType, typename IndexType, size_t PackSize, size_t num_dims,
         size_t rotary_emb_dim>
void LaunchKernel(ep::CudaStream* stream, const T* x, const T* cos, const T* sin,
                  const PositionType* position_ids, T* out, const int64_t* position_shape,
                  const std::string& x_layout, const std::string& output_layout,
                  const std::string& mode, const T theta, const IndexType rotary_size,
                  const IndexType b, const IndexType m, const IndexType h, const IndexType k,
                  const IndexType x_b_stride, const IndexType x_m_stride,
                  const IndexType x_h_stride, const IndexType x_offset,
                  const IndexType out_b_stride, const IndexType out_m_stride,
                  const IndexType out_h_stride, IndexType num_elements) {
  const IndexType k0 = rotary_size / rotary_emb_dim,
                  k1 = rotary_size;  // TODO: this only support 1d, 2d, rotary postional encoding

  const IndexType rotate_stride = rotary_size / (2 * rotary_emb_dim);

  const IndexType actual_rotary_size = rotary_size / rotary_emb_dim;
  const float inv_actual_rotary_size = 1.0 / actual_rotary_size;

  struct FusedApplyRotaryEmbParam<T, PositionType, IndexType, num_dims, rotary_emb_dim> param(
      x, cos, sin, position_ids, out, theta, inv_actual_rotary_size, actual_rotary_size,
      rotary_size, rotate_stride, num_elements, k, k0, k1, x_offset);

  const IndexType ref_strides[num_dims] = {m * h * k, h * k, k, 1};
  const IndexType out_strides[num_dims] = {out_b_stride, out_m_stride, out_h_stride, 1};
  const IndexType x_strides[num_dims] = {x_b_stride, x_m_stride, x_h_stride, 1};

  param.sinuous_m_stride = actual_rotary_size;

  const IndexType position_m = position_shape ? static_cast<IndexType>(position_shape[2]) : m;
  param.position_rotate_stride = position_m;
  param.position_b_stride = position_m * rotary_emb_dim;

// K has to be the last dimension, only k&m matters, therefore strides other than k&m does not
// really needs to be computed
#pragma unroll
  for (int i = 0; i < num_dims; i++) {
    param.ref_stride[i] = ref_strides[i];
    param.out_stride[i] = out_strides[i];
    param.x_stride[i] = x_strides[i];
  }

  constexpr size_t blk_size = 128;

  if (mode == "plane") {
    param.num_elements = param.num_elements * PackSize;
    PlaneKernel<T, PositionType, IndexType, num_dims, rotary_emb_dim>
        <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(
            param);
  } else {
    IntervalKernel<T, PositionType, IndexType, PackSize, num_dims, rotary_emb_dim>
        <<<(param.num_elements + blk_size - 1) / blk_size, blk_size, 0, stream->cuda_stream()>>>(
            param);
  }
}

template<typename T, typename PositionType, typename IndexType, size_t num_dims,
         size_t rotary_emb_dim>
void DispatchPackSize(ep::CudaStream* stream, const T* x, const T* cos, const T* sin,
                      const PositionType* position_ids, T* out, const int64_t* position_shape,
                      const std::string& x_layout, const std::string& output_layout,
                      const std::string& mode, const T theta, const IndexType rotary_size,
                      const IndexType b, const IndexType m, const IndexType h, const IndexType k,
                      const IndexType x_b_stride, const IndexType x_m_stride,
                      const IndexType x_h_stride, const IndexType x_offset,
                      const IndexType out_b_stride, const IndexType out_m_stride,
                      const IndexType out_h_stride, IndexType num_elements) {
  const auto CheckPackSize = [&](const size_t PackSize) {
    bool r = (((reinterpret_cast<uintptr_t>(x) % (sizeof(T) * PackSize)) == 0)
              && (((rotary_size / rotary_emb_dim) % PackSize) == 0)
              && (((k - rotary_size) % PackSize) == 0) && ((16 / sizeof(T)) >= PackSize));
    return r;
  };

  if (CheckPackSize(8)) {
    num_elements /= 8;
    LaunchKernel<T, PositionType, IndexType, 8, num_dims, rotary_emb_dim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, x_b_stride, x_m_stride, x_h_stride, x_offset, out_b_stride,
        out_m_stride, out_h_stride, num_elements);
  } else if (CheckPackSize(4)) {
    num_elements /= 4;
    LaunchKernel<T, PositionType, IndexType, 4, num_dims, rotary_emb_dim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, x_b_stride, x_m_stride, x_h_stride, x_offset, out_b_stride,
        out_m_stride, out_h_stride, num_elements);
  } else {
    num_elements /= 2;
    LaunchKernel<T, PositionType, IndexType, 2, num_dims, rotary_emb_dim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, x_b_stride, x_m_stride, x_h_stride, x_offset, out_b_stride,
        out_m_stride, out_h_stride, num_elements);
  }
}

template<typename T, typename PositionType, size_t num_dims, size_t rotary_emb_dim>
void DispatchIndex(ep::CudaStream* stream, const T* x, const T* cos, const T* sin,
                   const PositionType* position_ids, T* out, const int64_t* position_shape,
                   const std::string& x_layout, const std::string& output_layout,
                   const std::string& mode, const T theta, const int64_t rotary_size,
                   const int64_t b, const int64_t m, const int64_t h, const int64_t k,
                   const int64_t x_b_stride, const int64_t x_m_stride, const int64_t x_h_stride,
                   const int64_t x_offset, const int64_t out_b_stride, const int64_t out_m_stride,
                   const int64_t out_h_stride) {
  int64_t num_elements = b * m * h * k;
  if (num_elements < (1 << 30)) {
    DispatchPackSize<T, PositionType, int32_t, num_dims, rotary_emb_dim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, static_cast<int32_t>(rotary_size), static_cast<int32_t>(b), static_cast<int32_t>(m),
        static_cast<int32_t>(h), static_cast<int32_t>(k), static_cast<int32_t>(x_b_stride),
        static_cast<int32_t>(x_m_stride), static_cast<int32_t>(x_h_stride),
        static_cast<int32_t>(x_offset), static_cast<int32_t>(out_b_stride),
        static_cast<int32_t>(out_m_stride), static_cast<int32_t>(out_h_stride),
        static_cast<int32_t>(num_elements));
  } else {
    DispatchPackSize<T, PositionType, int64_t, num_dims, rotary_emb_dim>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, x_b_stride, x_m_stride, x_h_stride, x_offset, out_b_stride,
        out_m_stride, out_h_stride, num_elements);
  }
}

template<typename T, typename PositionType, size_t num_dims>
void DispatchRotaryEmbeddingDimension(ep::CudaStream* stream, const T* x, const T* cos,
                                      const T* sin, const PositionType* position_ids, T* out,
                                      const int64_t* position_shape, const std::string& x_layout,
                                      const std::string& output_layout, const std::string& mode,
                                      const T theta, const int64_t rotary_size,
                                      const int rotary_emb_dim, const int64_t b, const int64_t m,
                                      const int64_t h, const int64_t k, const int64_t x_b_stride,
                                      const int64_t x_m_stride, const int64_t x_h_stride,
                                      const int64_t x_offset, const int64_t out_b_stride,
                                      const int64_t out_m_stride, const int64_t out_h_stride) {
  if (rotary_emb_dim == 1) {
    DispatchIndex<T, PositionType, num_dims, 1>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, x_b_stride, x_m_stride, x_h_stride, x_offset, out_b_stride,
        out_m_stride, out_h_stride);
  } else if (rotary_emb_dim == 2) {
    DispatchIndex<T, PositionType, num_dims, 2>(
        stream, x, cos, sin, position_ids, out, position_shape, x_layout, output_layout, mode,
        theta, rotary_size, b, m, h, k, x_b_stride, x_m_stride, x_h_stride, x_offset, out_b_stride,
        out_m_stride, out_h_stride);
  }
}

template<typename T, typename PositionType>
class FusedApplyRotaryEmbKernel final : public user_op::OpKernel {
 public:
  FusedApplyRotaryEmbKernel() = default;
  ~FusedApplyRotaryEmbKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* cos = nullptr;
    user_op::Tensor* sin = nullptr;
    user_op::Tensor* position_ids = nullptr;
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const std::string& x_layout = ctx->Attr<std::string>("x_layout");
    const std::string& output_layout = ctx->Attr<std::string>("output_layout");
    const std::string& mode = ctx->Attr<std::string>("mode");
    const int64_t tensor_index = ctx->Attr<int64_t>("tensor_index");
    const int64_t k_size = ctx->Attr<int64_t>("k_size");
    const int64_t rotary_size = ctx->Attr<int64_t>("rotary_size");
    const float theta = 1.0f / ctx->Attr<float>("base");
    int rotary_emb_dim = 1;

    if (ctx->has_input("cos", 0)) { cos = ctx->Tensor4ArgNameAndIndex("cos", 0); }

    if (ctx->has_input("sin", 0)) { sin = ctx->Tensor4ArgNameAndIndex("sin", 0); }

    if (ctx->has_input("position_ids", 0)) {
      position_ids = ctx->Tensor4ArgNameAndIndex("position_ids", 0);
      rotary_emb_dim = position_ids->shape_view().At(1);
    }

    constexpr size_t ndims = 4;
    int64_t b = 0;
    int64_t m = 0;
    int64_t h = 0;
    int64_t k = 0;
    int64_t out_b_stride = 0, out_m_stride = 0, out_h_stride = 0, out_offset = 0;
    int64_t x_b_stride = 0, x_m_stride = 0, x_h_stride = 0, x_offset = 0;

    ParseDims(out->shape_view(), output_layout, Optional<int64_t>(), k_size, 0, &b, &m, &h, &k,
              &out_b_stride, &out_m_stride, &out_h_stride, &out_offset);
    ParseDims(x->shape_view(), x_layout, Optional<int64_t>(), k_size, tensor_index, &b, &m, &h, &k,
              &x_b_stride, &x_m_stride, &x_h_stride, &x_offset);

    // TODO: hard code num_dims & seems redundant template problem...
    DispatchRotaryEmbeddingDimension<T, PositionType, ndims>(
        ctx->stream()->As<ep::CudaStream>(), reinterpret_cast<const T*>(x->dptr()),
        cos ? reinterpret_cast<const T*>(cos->dptr()) : nullptr,
        sin ? reinterpret_cast<const T*>(sin->dptr()) : nullptr,
        position_ids ? reinterpret_cast<const PositionType*>(position_ids->dptr()) : nullptr,
        reinterpret_cast<T*>(out->mut_dptr()),
        position_ids ? position_ids->shape_view().data() : nullptr, x_layout, output_layout, mode,
        static_cast<T>(theta), rotary_size, rotary_emb_dim, b, m, h, k, x_b_stride, x_m_stride,
        x_h_stride, x_offset, out_b_stride, out_m_stride, out_h_stride);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(dtype, position_type)          \
  REGISTER_USER_KERNEL("fused_apply_rotary_emb")                           \
      .SetCreateFn<FusedApplyRotaryEmbKernel<dtype, position_type>>()      \
      .SetIsMatchedHob(                                                    \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                  \
          && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
          && (user_op::HobInputSize("position_ids") == 1)                  \
          && (user_op::HobDataType("position_ids", 0) == GetDataType<position_type>::value));

#define REGISTER_FUSED_APPLY_ROTARY_EMB_GPU_DTYPE(dtype)                                \
  REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(dtype, int64_t);                                  \
  REGISTER_FUSED_APPLY_ROTARY_EMB_GPU(dtype, int32_t);                                  \
  REGISTER_USER_KERNEL("fused_apply_rotary_emb")                                        \
      .SetCreateFn<FusedApplyRotaryEmbKernel<dtype, int64_t>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobInputSize("position_ids") == 0));

REGISTER_FUSED_APPLY_ROTARY_EMB_GPU_DTYPE(float);
REGISTER_FUSED_APPLY_ROTARY_EMB_GPU_DTYPE(half);
#if CUDA_VERSION >= 11000
REGISTER_FUSED_APPLY_ROTARY_EMB_GPU_DTYPE(nv_bfloat16);
#endif  // CUDA_VERSION >= 11000

}  // namespace

}  // namespace user_op

}  // namespace oneflow

#endif  // WITH_CUTLASS
