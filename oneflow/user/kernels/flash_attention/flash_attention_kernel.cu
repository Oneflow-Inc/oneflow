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
#include "fmha.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

// copy from
void set_params_fprop(FMHA_fprop_params& params,
                      // sizes
                      const size_t b, const size_t seqlen_q, const size_t seqlen_k,
                      const size_t num_head, const size_t head_size, const size_t q_row_stride,
                      const size_t k_row_stride, const size_t v_row_stride,
                      const size_t q_head_stride, const size_t k_head_stride,
                      const size_t v_head_stride,
                      // device pointers
                      void* q_ptr, void* k_ptr, void* v_ptr, void* cu_seqlens_q_d,
                      void* cu_seqlens_k_d, void* o_packed_d, void* o_tmp_d, void* s_d,
                      void* softmax_lse_d, float p_dropout, float softmax_scale, bool is_causal) {
  Data_type data_type = DATA_TYPE_FP16;
  // Reset the parameters
  memset(&params, 0, sizeof(params));

  params.is_bf16 = false;

  // Set the pointers and strides.
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.q_row_stride_in_elts = q_row_stride;
  params.k_row_stride_in_elts = k_row_stride;
  params.v_row_stride_in_elts = v_row_stride;
  params.q_head_stride_in_elts = q_head_stride;
  params.k_head_stride_in_elts = k_head_stride;
  params.v_head_stride_in_elts = v_head_stride;
  params.o_ptr = o_packed_d;
  params.o_row_stride_in_elts = num_head * head_size;
  params.o_head_stride_in_elts = head_size;
  params.o_tmp_ptr = o_tmp_d;

  params.cu_seqlens_q = static_cast<int32_t*>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int32_t*>(cu_seqlens_k_d);

  // S = softmax(P)
  params.s_ptr = s_d;
  params.s_stride_in_bytes = get_size_in_bytes(b * num_head * seqlen_k, data_type);

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = num_head;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.d = head_size;

  // Set the different scale values.
  // const float scale_bmm1 = 1.f / sqrtf(d);
  const float scale_bmm1 = softmax_scale;

  params.scale_bmm1f = scale_bmm1;
  set_alpha(params.scale_bmm1, scale_bmm1, data_type);

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_bmm1_rp_dropout = params.rp_dropout * params.scale_bmm1f;
  set_alpha(params.scale_dropout, params.rp_dropout, data_type);

  params.is_causal = is_causal;
}

void set_params_dgrad(FMHA_dgrad_params& params,
                      // sizes
                      const size_t b, const size_t seqlen_q, const size_t seqlen_k,
                      const size_t num_head, const size_t head_size, const size_t q_row_stride,
                      const size_t k_row_stride, const size_t v_row_stride,
                      const size_t q_head_stride, const size_t k_head_stride,
                      const size_t v_head_stride,
                      // device pointers
                      void* q_ptr, void* k_ptr, void* v_ptr, void* dq_ptr, void* dk_ptr,
                      void* dv_ptr, void* cu_seqlens_q_d, void* cu_seqlens_k_d, void* o_packed_d,
                      void* dq_tmp_d, void* do_packed_d, void* softmax_lse_d, void* dsoftmax_sum_d,
                      float p_dropout, float softmax_scale, bool is_causal) {
  set_params_fprop(params, b, seqlen_q, seqlen_k, num_head, head_size, q_row_stride, k_row_stride,
                   v_row_stride, q_head_stride, k_head_stride, v_head_stride, q_ptr, k_ptr, v_ptr,
                   cu_seqlens_q_d, cu_seqlens_k_d, o_packed_d,
                   dq_tmp_d,  // Reusing the o_tmp_ptr variable to store dq_tmp
                   nullptr, softmax_lse_d, p_dropout, softmax_scale, is_causal);

  // Set the pointers and strides.
  params.dq_ptr = dq_ptr;
  params.dk_ptr = dk_ptr;
  params.dv_ptr = dv_ptr;
  params.dq_row_stride_in_elts = q_row_stride;
  params.dk_row_stride_in_elts = k_row_stride;
  params.dv_row_stride_in_elts = v_row_stride;
  params.dq_head_stride_in_elts = q_head_stride;
  params.dk_head_stride_in_elts = k_head_stride;
  params.dv_head_stride_in_elts = v_head_stride;
  params.do_ptr = do_packed_d;

  // Softmax sum
  params.dsoftmax_sum = dsoftmax_sum_d;
}

}  // namespace

class FlashAttentionKernel final : public user_op::OpKernel {
 public:
  FlashAttentionKernel() = default;
  ~FlashAttentionKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const user_op::Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const user_op::Tensor* valid_seqlens_q = ctx->Tensor4ArgNameAndIndex("valid_seqlens_q", 0);
    const user_op::Tensor* valid_seqlens_k = ctx->Tensor4ArgNameAndIndex("valid_seqlens_k", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* softmax_lse = ctx->Tensor4ArgNameAndIndex("softmax_lse", 0);
    const bool is_causal = ctx->Attr<bool>("causal");
    const float dropout_rate = ctx->Attr<float>("dropout_rate");
    bool is_dropout = dropout_rate > 0.0;
    LOG(ERROR) << "is_dropout" << is_dropout;
    cudaStream_t stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    const cudaDeviceProp& device_props = ctx->stream()->As<ep::CudaStream>()->device_properties();
    Launch_params<FMHA_fprop_params> launch_params(&device_props, stream, is_dropout,
                                                   /*return_softmax*/ false);
    const int64_t batch_size = query->shape_view().At(0);
    const int64_t max_seqlen_q = query->shape_view().At(1);
    const int64_t max_seqlen_k = key->shape_view().At(1);
    const int64_t num_head = query->shape_view().At(2);
    const int64_t head_size = query->shape_view().At(3);
    const size_t row_stride = num_head * head_size;
    const size_t head_stride = head_size;
    const float softmax_scale = 1.f / sqrtf(head_size);  // attr?

    bool is_sm75 = device_props.major == 7 && device_props.minor == 5;
    bool is_sm80 = device_props.major == 8 && device_props.minor == 0;
    bool is_sm8x = device_props.major == 8 && device_props.minor >= 0;
    int blocksize_c = ((head_size == 128 && (is_dropout || !is_sm80))
                       || (is_sm75 && head_size == 64 && is_dropout))
                          ? 128
                          : 256;
    bool loop = max_seqlen_k > blocksize_c;

    set_params_fprop(launch_params.params, batch_size, max_seqlen_q, max_seqlen_k, num_head,
                     head_size, row_stride, row_stride, row_stride, head_stride, head_stride,
                     head_stride, const_cast<void*>(query->dptr()), const_cast<void*>(key->dptr()),
                     const_cast<void*>(value->dptr()), const_cast<void*>(valid_seqlens_q->dptr()),
                     const_cast<void*>(valid_seqlens_k->dptr()), out->mut_dptr(),
                     loop ? tmp_buffer->mut_dptr() : nullptr,
                     /*not return softmax*/ nullptr, softmax_lse->mut_dptr(), dropout_rate,
                     softmax_scale, is_causal);
    run_fmha_fp16_sm80(launch_params, /*configure=*/true);
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state

    // state
    int64_t counter_offset = launch_params.elts_per_thread;

    if (is_dropout) {
      one::CUDAGeneratorState* cuda_gen_state = cuda_generator->cuda_gen_state();
      launch_params.params.philox_args = PhiloxCudaState(this->seed_, offset);
    }

    run_fmha_fprop(launch_params, /*configure=*/false);
    run_fmha_fp16_sm80(launch_params, /*configure=*/false);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("flash_attention")
    .SetCreateFn<FlashAttentionKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("query", 0) == DataType::kFloat16
                         || user_op::HobDataType("query", 0) == DataType::kBFloat16))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const user_op::TensorDesc& softmax_lse = ctx->InputTensorDesc("softmax_lse", 0);
      const user_op::TensorDesc& out = ctx->InputTensorDesc("out", 0);
      return out.shape().elem_cnt() * GetSizeOfDataType(softmax_lse.data_type());
    });

class FlashAttentionGradKernel final : public user_op::OpKernel {
 public:
  FlashAttentionGradKernel() = default;
  ~FlashAttentionGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* out_grad = ctx->Tensor4ArgNameAndIndex("out_grad", 0);
    const user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const user_op::Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const user_op::Tensor* softmax_lse = ctx->Tensor4ArgNameAndIndex("softmax_lse", 0);
    const user_op::Tensor* valid_seqlens_q = ctx->Tensor4ArgNameAndIndex("valid_seqlens_q", 0);
    const user_op::Tensor* valid_seqlens_k = ctx->Tensor4ArgNameAndIndex("valid_seqlens_k", 0);
    user_op::Tensor* query_grad = ctx->Tensor4ArgNameAndIndex("query_grad", 0);
    user_op::Tensor* key_grad = ctx->Tensor4ArgNameAndIndex("key_grad", 0);
    user_op::Tensor* value_grad = ctx->Tensor4ArgNameAndIndex("value_grad", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const bool is_causal = ctx->Attr<bool>("causal");
    const float dropout_rate = ctx->Attr<float>("dropout_rate");
    bool is_dropout = dropout_rate > 0.0;
    cudaStream_t stream = ctx->stream()->As<ep::CudaStream>()->cuda_stream();
    const cudaDeviceProp& device_props = ctx->stream()->As<ep::CudaStream>()->device_properties();
    Launch_params<FMHA_dgrad_params> launch_params(&device_props, stream, is_dropout,
                                                   /*return_softmax*/ false);
    const int64_t batch_size = query->shape_view().At(0);
    const int64_t max_seqlen_q = query->shape_view().At(1);
    const int64_t max_seqlen_k = key->shape_view().At(1);
    const int64_t num_head = query->shape_view().At(2);
    const int64_t head_size = query->shape_view().At(3);
    const size_t row_stride = num_head * head_size;
    const size_t head_stride = head_size;
    const float softmax_scale = 1.f / sqrtf(head_size);  // attr?

    bool is_sm75 = device_props.major == 7 && device_props.minor == 5;
    bool is_sm80 = device_props.major == 8 && device_props.minor == 0;
    bool is_sm8x = device_props.major == 8 && device_props.minor >= 0;
    int blocksize_c = ((head_size == 128 && (is_dropout || !is_sm80))
                       || (is_sm75 && head_size == 64 && is_dropout))
                          ? 128
                          : 256;
    bool loop = max_seqlen_k > blocksize_c;
    void* query_grad_tmp_ptr = tmp_buffer->mut_dptr();
    DataType tmp_buf_type = softmax_lse->data_type();
    size_t query_grad_tmp_bytes =
        GetCudaAlignedSize(query_grad->shape_view().elem_cnt() * GetSizeOfDataType(tmp_buf_type));
    // why dsoftmax_sum?
    void* dsoftmax_sum_ptr =
        reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + query_grad_tmp_bytes);
    set_params_dgrad(launch_params.params, batch_size, max_seqlen_q, max_seqlen_k, num_head,
                     head_size, row_stride, row_stride, row_stride, head_stride, head_stride,
                     head_stride, const_cast<void*>(query->dptr()), const_cast<void*>(key->dptr()),
                     const_cast<void*>(value->dptr()), query_grad->mut_dptr(), key_grad->mut_dptr(),
                     value_grad->mut_dptr(), const_cast<void*>(valid_seqlens_q->dptr()),
                     const_cast<void*>(valid_seqlens_k->dptr()), const_cast<void*>(out->dptr()),
                     loop ? query_grad_tmp_ptr : nullptr, const_cast<void*>(out_grad->dptr()),
                     const_cast<void*>(softmax_lse->dptr()), dsoftmax_sum_ptr, dropout_rate,
                     softmax_scale, is_causal);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    int64_t counter_offset = launch_params.elts_per_thread;
    at::PhiloxCudaState rng_engine_inputs;

    if (is_dropout) {
      // See Note [Acquire lock when using random generators]
      // std::lock_guard<std::mutex> lock(gen->mutex_);
      // launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
      // TODO:
      uint64_t seed = 0;
      uint64_t offset = 0;
      launch_params.params.philox_args = at::PhiloxCudaState(seed, offset);
    }
    run_fmha_dgrad_fp16_sm80(launch_params, stream);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("flash_attention_grad")
    .SetCreateFn<FlashAttentionGradKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobDataType("query", 0) == DataType::kFloat16
                         || user_op::HobDataType("query", 0) == DataType::kBFloat16))
    .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
      const user_op::TensorDesc& query_grad = ctx->OutputTensorDesc("query_grad", 0);
      const user_op::TensorDesc& softmax_lse = ctx->InputTensorDesc("softmax_lse", 0);
      size_t buffer_dtype_bytes = GetSizeOfDataType(softmax_lse.data_type());
      size_t query_grad_tmp_bytes =
          GetCudaAlignedSize(query_grad.shape().elem_cnt() * buffer_dtype_bytes);
      size_t dsoftmax_sum_bytes =
          GetCudaAlignedSize(softmax_lse.shape().elem_cnt() * buffer_dtype_bytes);
      return query_grad_tmp_bytes + dsoftmax_sum_bytes;
    });

}  // namespace oneflow
