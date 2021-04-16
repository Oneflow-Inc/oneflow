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
#include "oneflow/core/cuda/bgemm.h"
#include "oneflow/user/kernels/slice_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
void bgemm(DeviceCtx* ctx, char opa, char opb, int64_t m, int64_t n, int64_t k, float alpha,
           const T* a, int64_t lda, int64_t stridea, const T* b, int64_t ldb, int64_t strideb,
           float beta, T* c, int64_t ldc, int64_t stridec, int64_t batch_size) {
  // swap m and n, a and b to convert from row-major to col-major
  cuda::blas::bgemm<T>(ctx->cublas_pmd_handle(), opb, opa, n, m, k, static_cast<T>(alpha), b, ldb,
                       strideb, a, lda, stridea, static_cast<T>(beta), c, ldc, stridec, batch_size);
}

SliceParams ConstructSliceParams4Value(int64_t seq_len, int64_t batch_size, int64_t num_heads,
                                       int64_t head_size) {
  // slice (s, b, n, 3, h) to (s, b, n, 1, h)
  SliceParams params;
  std::memset(&params, 0, sizeof(SliceParams));
  params.ndim = 4;
  params.dims[0] = seq_len;
  params.dims[1] = batch_size;
  params.dims[2] = num_heads;
  params.dims[3] = 3 * head_size;
  params.start[0] = 0;
  params.start[1] = 0;
  params.start[2] = 0;
  params.start[3] = 2 * head_size;
  params.step[0] = 1;
  params.step[1] = 1;
  params.step[2] = 1;
  params.step[3] = 1;
  params.size[0] = seq_len;
  params.size[1] = batch_size;
  params.size[2] = num_heads;
  params.size[3] = head_size;
  return params;
}

template<typename T>
void transpose(DeviceCtx* ctx, const user_op::Tensor* in, const std::vector<int32_t>& perm,
               user_op::Tensor* out) {
  using PackType = int64_t;
  constexpr size_t pack_size = sizeof(PackType) / sizeof(T);
  const ShapeView& in_shape = in->shape();
  const ShapeView& out_shape = out->shape();
  int64_t in_last_dim = in_shape.At(in_shape.NumAxes() - 1);
  int64_t out_last_dim = out_shape.At(out_shape.NumAxes() - 1);
  if (pack_size != 1 && perm.back() == perm.size() - 1 && in_last_dim % pack_size == 0) {
    CHECK_EQ(in_last_dim, out_last_dim);
    DimVector packed_in_dim_vec;
    in_shape.ToDimVector(&packed_in_dim_vec);
    packed_in_dim_vec.back() /= pack_size;
    const Shape packed_in_shape(packed_in_dim_vec);
    DimVector packed_out_dim_vec;
    out_shape.ToDimVector(&packed_out_dim_vec);
    packed_out_dim_vec.back() /= pack_size;
    const Shape packed_out_shape(packed_out_dim_vec);
    NewKernelUtil<DeviceType::kGPU>::Transpose(ctx, packed_in_shape.NumAxes(), packed_in_shape,
                                               packed_out_shape, perm, packed_in_shape.elem_cnt(),
                                               reinterpret_cast<const PackType*>(in->dptr<T>()),
                                               reinterpret_cast<PackType*>(out->mut_dptr<T>()));
  } else {
    NewKernelUtil<DeviceType::kGPU>::Transpose(ctx, in_shape.NumAxes(), in_shape, out_shape, perm,
                                               in_shape.elem_cnt(), in->dptr<T>(),
                                               out->mut_dptr<T>());
  }
}

template<typename T>
class FusedSelfAttentionQueryMulKeyAndValueGpuKernel final : public user_op::OpKernel {
 public:
  FusedSelfAttentionQueryMulKeyAndValueGpuKernel() = default;
  ~FusedSelfAttentionQueryMulKeyAndValueGpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* h_tensor = ctx->Tensor4ArgNameAndIndex("hidden_states", 0);
    int64_t seq_len = h_tensor->shape().At(0);
    int64_t batch_size = h_tensor->shape().At(1);
    int64_t hidden_size = h_tensor->shape().At(2);
    int64_t head_size = ctx->Attr<int64_t>("head_size");
    int64_t ld = batch_size * hidden_size;
    int64_t stride = 3 * head_size;
    int64_t num_heads = hidden_size / stride;

    // q * k: (s, b, n, h) x (s, b, n, h) => (b, n, s, h) x (b, n, s, h)
    // => (b, n, s, h) x (b, n, h, s) -> (b, n, s, s)
    float alpha = ctx->Attr<float>("alpha");
    user_op::Tensor* qmk_tensor = ctx->Tensor4ArgNameAndIndex("query_mul_key", 0);
    bgemm<T>(ctx->device_ctx(), 'N', 'T', seq_len, seq_len, head_size, alpha, h_tensor->dptr<T>(),
             ld, stride, h_tensor->dptr<T>(), ld, stride, 0.0f, qmk_tensor->mut_dptr<T>(), seq_len,
             seq_len * seq_len, batch_size * num_heads);

    // slice v
    user_op::Tensor* tmp_v_tensor = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* v_tensor = ctx->Tensor4ArgNameAndIndex("value", 0);
    SliceParams params = ConstructSliceParams4Value(seq_len, batch_size, num_heads, head_size);
    SliceKernelUtil<DeviceType::kGPU, T>::Forward(ctx->device_ctx(), params, h_tensor->dptr<T>(),
                                                  tmp_v_tensor->mut_dptr<T>());
    // v from (s, b, n, h) transpose to (b, n, s, h)
    transpose<T>(ctx->device_ctx(), tmp_v_tensor, {1, 2, 0, 3}, v_tensor);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class FusedSelfAttentionQueryMulKeyAndValueGradGpuKernel final : public user_op::OpKernel {
 public:
  FusedSelfAttentionQueryMulKeyAndValueGradGpuKernel() = default;
  ~FusedSelfAttentionQueryMulKeyAndValueGradGpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* h_grad_tensor = ctx->Tensor4ArgNameAndIndex("hidden_states_grad", 0);
    int64_t seq_len = h_grad_tensor->shape().At(0);
    int64_t batch_size = h_grad_tensor->shape().At(1);
    int64_t hidden_size = h_grad_tensor->shape().At(2);
    int64_t head_size = ctx->Attr<int64_t>("head_size");
    int64_t ld = batch_size * hidden_size;
    int64_t stride = 3 * head_size;
    int64_t num_heads = hidden_size / stride;

    const user_op::Tensor* v_grad_tensor = ctx->Tensor4ArgNameAndIndex("value_grad", 0);
    user_op::Tensor* tmp_v_tensor = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    // transpose from (b, n, s, h) to (s, b, n, h)
    transpose<T>(ctx->device_ctx(), v_grad_tensor, {1, 2, 0, 3}, tmp_v_tensor);
    // slice v grad
    SliceParams params = ConstructSliceParams4Value(seq_len, batch_size, num_heads, head_size);
    SliceKernelUtil<DeviceType::kGPU, T>::Backward(
        ctx->device_ctx(), params, tmp_v_tensor->dptr<T>(), h_grad_tensor->mut_dptr<T>());

    float alpha = ctx->Attr<float>("alpha");
    const user_op::Tensor* qmk_grad_tensor = ctx->Tensor4ArgNameAndIndex("query_mul_key_grad", 0);
    const user_op::Tensor* h_tensor = ctx->Tensor4ArgNameAndIndex("hidden_states", 0);
    // grad_q = grad_qmk * k
    // (b, n, sq, sk) x (b, n, sk, h) -> (b, n, s, h) <= (s, b, n, h) <= (s, b, n, 3, h)
    bgemm<T>(ctx->device_ctx(), 'N', 'N', seq_len, head_size, seq_len, alpha,
             qmk_grad_tensor->dptr<T>(), seq_len, seq_len * seq_len, h_tensor->dptr<T>(), ld,
             stride, 0.0f, h_grad_tensor->mut_dptr<T>(), ld, stride, batch_size * num_heads);
    // grad_k = grad_qmk * q
    // (b, n, sk, sq) x (b, n, sq, h) -> (b, n, sk, h) <= (s, b, n, h) <= (s, b, n, 3, h)
    bgemm<T>(ctx->device_ctx(), 'T', 'N', seq_len, head_size, seq_len, alpha,
             qmk_grad_tensor->dptr<T>(), seq_len, seq_len * seq_len, h_tensor->dptr<T>(), ld,
             stride, 0.0f, h_grad_tensor->mut_dptr<T>(), ld, stride, batch_size * num_heads);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

size_t InferTmpBufferSize(user_op::InferContext* ctx) {
  const Shape* value_shape = ctx->Shape4ArgNameAndIndex("value", 0);
  DataType value_dtype = *ctx->Dtype4ArgNameAndIndex("value", 0);
  return value_shape->elem_cnt() * GetSizeOfDataType(value_dtype);
}

size_t InferGradTmpBufferSize(user_op::InferContext* ctx) {
  const Shape* value_shape = ctx->Shape4ArgNameAndIndex("value_grad", 0);
  DataType value_dtype = *ctx->Dtype4ArgNameAndIndex("value_grad", 0);
  return value_shape->elem_cnt() * GetSizeOfDataType(value_dtype);
}

}  // namespace

#define REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GPU_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("fused_self_attention_query_mul_key_and_value")                            \
      .SetCreateFn<FusedSelfAttentionQueryMulKeyAndValueGpuKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                              \
                       & (user_op::HobDataType("hidden_states", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTmpBufferSize);

#define REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GRAD_GPU_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("fused_self_attention_query_mul_key_and_value_grad")                       \
      .SetCreateFn<FusedSelfAttentionQueryMulKeyAndValueGradGpuKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                              \
                       & (user_op::HobDataType("hidden_states", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferGradTmpBufferSize);

REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GPU_KERNEL(float)
REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GPU_KERNEL(float16)
REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GRAD_GPU_KERNEL(float)
REGISTER_FUSED_SELF_ATTENTION_QUERY_MUL_KEY_AND_VALUE_GRAD_GPU_KERNEL(float16)

}  // namespace oneflow
