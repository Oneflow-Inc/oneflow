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
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/framework/random_generator_impl.h"

namespace oneflow {

namespace {

// copy from
void set_params_fprop(FMHA_fprop_params& params, bool is_bf16,
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

  params.is_bf16 = is_bf16;

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

void set_params_dgrad(FMHA_dgrad_params& params, bool is_bf16,
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
  set_params_fprop(params, is_bf16, b, seqlen_q, seqlen_k, num_head, head_size, q_row_stride, k_row_stride,
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

class FlashAttentionKernelState : public user_op::OpKernelState {
 public:
  explicit FlashAttentionKernelState(const std::shared_ptr<one::Generator>& generator)
      : generator_(generator), offset_(0) {}

  const std::shared_ptr<one::Generator>& generator() const { return generator_; }

  int64_t offset(int64_t elem_per_thread) {
    int64_t cur_offset = offset_;
    offset_ += elem_per_thread;
    return cur_offset;
  }

 private:
  std::shared_ptr<one::Generator> generator_;
  int64_t offset_;
};

}  // namespace

class FlashAttentionKernel final : public user_op::OpKernel {
 public:
  FlashAttentionKernel() = default;
  ~FlashAttentionKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA));
    return std::make_shared<FlashAttentionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const user_op::Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const user_op::Tensor* cu_seqlens_q = ctx->Tensor4ArgNameAndIndex("cu_seqlens_q", 0);
    const user_op::Tensor* cu_seqlens_k = ctx->Tensor4ArgNameAndIndex("cu_seqlens_k", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* softmax_lse = ctx->Tensor4ArgNameAndIndex("softmax_lse", 0);
    // Note: deafult should be 1.f / sqrtf(head_size)
    const float softmax_scale = ctx->Attr<float>("softmax_scale");
    const bool is_causal = ctx->Attr<bool>("causal");
    const float dropout_rate = ctx->Attr<float>("dropout_rate");
    bool is_dropout = dropout_rate > 0.0;
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

    bool is_sm75 = device_props.major == 7 && device_props.minor == 5;
    bool is_sm80 = device_props.major == 8 && device_props.minor == 0;
    bool is_sm8x = device_props.major == 8 && device_props.minor >= 0;
    int blocksize_c = ((head_size == 128 && (is_dropout || !is_sm80))
                       || (is_sm75 && head_size == 64 && is_dropout))
                          ? 128
                          : 256;
    bool loop = max_seqlen_k > blocksize_c;

    const bool is_bf16 = (query->data_type() == DataType::kBFloat16);
    set_params_fprop(launch_params.params, is_bf16, batch_size, max_seqlen_q, max_seqlen_k, num_head,
                     head_size, row_stride, row_stride, row_stride, head_stride, head_stride,
                     head_stride, const_cast<void*>(query->dptr()), const_cast<void*>(key->dptr()),
                     const_cast<void*>(value->dptr()), const_cast<void*>(cu_seqlens_q->dptr()),
                     const_cast<void*>(cu_seqlens_k->dptr()), out->mut_dptr(),
                     loop ? tmp_buffer->mut_dptr() : nullptr,
                     /*not return softmax*/ nullptr, softmax_lse->mut_dptr(), dropout_rate,
                     softmax_scale, is_causal);
    run_fmha_fp16_sm80(launch_params, /*configure=*/true);
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state

    // state
    int64_t counter_offset = launch_params.elts_per_thread;

    if (is_dropout) {
      auto* kernel_state = dynamic_cast<FlashAttentionKernelState*>(state);
      CHECK_NOTNULL(kernel_state);
      const auto& generator = kernel_state->generator();
      CHECK_NOTNULL(generator);
      const auto device_index = ctx->stream()->device()->device_index();
      std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
          CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));
      uint64_t seed = cuda_generator->current_seed();
      uint64_t offset = kernel_state->offset(counter_offset);
      launch_params.params.philox_args = at::PhiloxCudaState(seed, offset);
    }

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

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA));
    return std::make_shared<FlashAttentionKernelState>(generator);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    const user_op::Tensor* out_grad = ctx->Tensor4ArgNameAndIndex("out_grad", 0);
    const user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* query = ctx->Tensor4ArgNameAndIndex("query", 0);
    const user_op::Tensor* key = ctx->Tensor4ArgNameAndIndex("key", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const user_op::Tensor* softmax_lse = ctx->Tensor4ArgNameAndIndex("softmax_lse", 0);
    const user_op::Tensor* cu_seqlens_q = ctx->Tensor4ArgNameAndIndex("cu_seqlens_q", 0);
    const user_op::Tensor* cu_seqlens_k = ctx->Tensor4ArgNameAndIndex("cu_seqlens_k", 0);
    user_op::Tensor* query_grad = ctx->Tensor4ArgNameAndIndex("query_grad", 0);
    user_op::Tensor* key_grad = ctx->Tensor4ArgNameAndIndex("key_grad", 0);
    user_op::Tensor* value_grad = ctx->Tensor4ArgNameAndIndex("value_grad", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    // Note: deafult should be 1.f / sqrtf(head_size)
    const float softmax_scale = ctx->Attr<float>("softmax_scale");
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
    const bool is_bf16 = (query->data_type() == DataType::kBFloat16);
    set_params_dgrad(launch_params.params, is_bf16, batch_size, max_seqlen_q, max_seqlen_k, num_head,
                     head_size, row_stride, row_stride, row_stride, head_stride, head_stride,
                     head_stride, const_cast<void*>(query->dptr()), const_cast<void*>(key->dptr()),
                     const_cast<void*>(value->dptr()), query_grad->mut_dptr(), key_grad->mut_dptr(),
                     value_grad->mut_dptr(), const_cast<void*>(cu_seqlens_q->dptr()),
                     const_cast<void*>(cu_seqlens_k->dptr()), const_cast<void*>(out->dptr()),
                     loop ? query_grad_tmp_ptr : nullptr, const_cast<void*>(out_grad->dptr()),
                     const_cast<void*>(softmax_lse->dptr()), dsoftmax_sum_ptr, dropout_rate,
                     softmax_scale, is_causal);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    int64_t counter_offset = launch_params.elts_per_thread;
    at::PhiloxCudaState rng_engine_inputs;

    if (is_dropout) {
      auto* kernel_state = dynamic_cast<FlashAttentionKernelState*>(state);
      CHECK_NOTNULL(kernel_state);
      const auto& generator = kernel_state->generator();
      CHECK_NOTNULL(generator);
      const auto device_index = ctx->stream()->device()->device_index();
      std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
          CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));
      uint64_t seed = cuda_generator->current_seed();
      uint64_t offset = kernel_state->offset(counter_offset);
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
