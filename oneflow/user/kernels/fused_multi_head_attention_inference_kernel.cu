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
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "kernel_forward.h"

namespace oneflow {

namespace user_op {

namespace {

struct Params {
  DataType data_type;
  int64_t num_batches;
  int64_t num_heads;
  int64_t query_seq_len;
  int64_t kv_seq_len;
  int64_t head_size;
  int64_t value_head_size;
  int64_t query_hidden_stride;
  int64_t key_hidden_stride;
  int64_t value_hidden_stride;
  bool causal;
  const void* query_ptr;
  const void* key_ptr;
  const void* value_ptr;
  void* out_ptr;
  void* workspace;
  int64_t workspace_size;
};

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block,
         bool single_value_iteration>
void LaunchCutlassFmha(const Params& params, ep::CudaStream* stream) {
  using Attention = AttentionKernel<T, ArchTag, is_aligned, queries_per_block, keys_per_block,
                                    single_value_iteration>;
  typename Attention::Params p;
  p.query_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.query_ptr));
  p.key_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.key_ptr));
  p.value_ptr = const_cast<T*>(reinterpret_cast<const T*>(params.value_ptr));
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
  p.causal = false;
  p.q_strideM = params.query_hidden_stride;
  p.k_strideM = params.key_hidden_stride;
  p.v_strideM = params.value_hidden_stride;

  p.q_strideH = params.head_size;
  p.k_strideH = params.head_size;
  p.v_strideH = params.value_head_size;
  p.o_strideH = params.query_seq_len * params.value_head_size;

  p.q_strideB = params.query_seq_len * p.q_strideM;
  p.k_strideB = params.kv_seq_len * p.k_strideM;
  p.v_strideB = params.kv_seq_len * p.v_strideM;
  p.o_strideB = p.o_strideH * p.num_heads;

  p.causal = params.causal;

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

template<typename T, typename ArchTag, bool is_aligned, int queries_per_block, int keys_per_block>
void DispatchSingleValueIteration(const Params& params, ep::CudaStream* stream) {
  if (params.value_head_size <= keys_per_block) {
    LaunchCutlassFmha<T, ArchTag, is_aligned, queries_per_block, keys_per_block, true>(params,
                                                                                       stream);
  } else {
    LaunchCutlassFmha<T, ArchTag, is_aligned, queries_per_block, keys_per_block, false>(params,
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
      && params.query_hidden_stride % (16 / sizeof(T)) == 0
      && params.key_hidden_stride % (16 / sizeof(T)) == 0) {
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
    DispatchArchTag<cutlass::tfloat32_t>(params, stream);
  } else {
    UNIMPLEMENTED();
  }
}

class FusedMultiHeadAttentionInferenceKernel final : public user_op::OpKernel {
 public:
  FusedMultiHeadAttentionInferenceKernel() = default;
  ~FusedMultiHeadAttentionInferenceKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const DataType data_type = query->data_type();
    CHECK_EQ(key->data_type(), data_type);
    CHECK_EQ(value->data_type(), data_type);
    CHECK_EQ(out->data_type(), data_type);
    CHECK_EQ(query->shape_view().NumAxes(), 3);
    CHECK_EQ(key->shape_view().NumAxes(), 3);
    CHECK_EQ(value->shape_view().NumAxes(), 3);
    CHECK_EQ(out->shape_view().NumAxes(), 3);
    const int64_t batch_size = query->shape_view().At(0);
    CHECK_EQ(key->shape_view().At(0), batch_size);
    CHECK_EQ(value->shape_view().At(0), batch_size);
    CHECK_EQ(out->shape_view().At(0), batch_size);
    const int64_t query_seq_len = query->shape_view().At(1);
    CHECK_EQ(out->shape_view().At(1), query_seq_len);
    const int64_t kv_seq_len = key->shape_view().At(1);
    CHECK_EQ(value->shape_view().At(1), kv_seq_len);
    const int64_t num_heads = ctx->Attr<int64_t>("num_heads");

    const auto ParseHiddenDim = [&](const std::string& tag, const ShapeView& shape,
                                    int64_t* hidden_slice_start, int64_t* hidden_size) {
      *hidden_slice_start = ctx->Attr<int64_t>(tag + "_hidden_slice_start");
      CHECK_GE(*hidden_slice_start, 0);
      int64_t hidden_slice_end = ctx->Attr<int64_t>(tag + "_hidden_slice_end");
      if (hidden_slice_end < 0) { hidden_slice_end = hidden_slice_end + shape.At(2) + 1; }
      CHECK_GT(hidden_slice_end, 0);
      CHECK_LE(hidden_slice_end, shape.At(2));
      CHECK_GT(hidden_slice_end, *hidden_slice_start);
      *hidden_size = hidden_slice_end - *hidden_slice_start;
      CHECK_EQ(*hidden_size % num_heads, 0);
    };

    int64_t query_hidden_offset = 0;
    int64_t query_hidden_size = 0;
    ParseHiddenDim("query", query->shape_view(), &query_hidden_offset, &query_hidden_size);

    int64_t key_hidden_offset = 0;
    int64_t key_hidden_size = 0;
    ParseHiddenDim("key", key->shape_view(), &key_hidden_offset, &key_hidden_size);
    CHECK_EQ(key_hidden_size, query_hidden_size);

    int64_t value_hidden_offset = 0;
    int64_t value_hidden_size = 0;
    ParseHiddenDim("value", value->shape_view(), &value_hidden_offset, &value_hidden_size);

    CHECK_EQ(out->shape_view().At(2), value_hidden_size);

    Params params{};
    params.data_type = data_type;
    params.num_batches = batch_size;
    params.num_heads = num_heads;
    params.query_seq_len = query_seq_len;
    params.kv_seq_len = kv_seq_len;
    params.head_size = query_hidden_size / num_heads;
    params.value_head_size = value_hidden_size / num_heads;
    params.query_hidden_stride = query->shape_view().At(2);
    params.key_hidden_stride = key->shape_view().At(2);
    params.value_hidden_stride = value->shape_view().At(2);
    params.query_ptr = query->dptr<char>() + query_hidden_offset;
    params.key_ptr = key->dptr<char>() + key_hidden_offset;
    params.value_ptr = value->dptr<char>() + value_hidden_offset;
    params.out_ptr = tmp->mut_dptr();
    const int64_t out_buffer_size =
        GetCudaAlignedSize(out->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
    const int64_t tmp_buffer_size = tmp->shape_view().elem_cnt();
    CHECK_GE(tmp_buffer_size, out_buffer_size);
    params.workspace = tmp->mut_dptr<char>() + out_buffer_size;
    params.workspace_size = tmp_buffer_size - out_buffer_size;
    params.causal = ctx->Attr<bool>("causal");

    DispatchCutlassFmha(params, ctx->stream()->As<ep::CudaStream>());

    auto permute =
        ep::primitive::NewPrimitive<ep::primitive::PermuteFactory>(ctx->device_type(), 4);
    CHECK(permute);
    const int64_t src_dims[4] = {batch_size, num_heads, query_seq_len,
                                 value_hidden_size / num_heads};
    const int permutation[4] = {0, 2, 1, 3};
    permute->Launch(ctx->stream(), data_type, 4, src_dims, tmp->dptr(), permutation,
                    out->mut_dptr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t InferTmpBufferSize(InferContext* ctx) {
  const auto& out_desc = ctx->OutputTensorDesc("out", 0);
  size_t buffer_size = 0;
  buffer_size +=
      GetCudaAlignedSize(out_desc.shape().elem_cnt() * GetSizeOfDataType(out_desc.data_type()));
  buffer_size +=
      GetCudaAlignedSize(out_desc.shape().elem_cnt() * GetSizeOfDataType(DataType::kFloat));
  return buffer_size;
}

}  // namespace

#define REGISTER_FUSED_MULTI_HEAD_ATTENTION_INFERENCE_KERNEL(dtype)    \
  REGISTER_USER_KERNEL("fused_multi_head_attention_inference")         \
      .SetCreateFn<FusedMultiHeadAttentionInferenceKernel>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == dtype))   \
      .SetInferTmpSizeFn(InferTmpBufferSize);

REGISTER_FUSED_MULTI_HEAD_ATTENTION_INFERENCE_KERNEL(DataType::kFloat16)
REGISTER_FUSED_MULTI_HEAD_ATTENTION_INFERENCE_KERNEL(DataType::kFloat)

}  // namespace user_op

}  // namespace oneflow
