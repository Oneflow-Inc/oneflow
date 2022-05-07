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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/user/kernels/model_update_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename G, typename IDX>
__global__ void SGDUpdateKernel(const int64_t embedding_size, T scale, float l1, float l2,
                                float weight_decay, const IDX* num_unique_ids,
                                const float* learning_rate, const T* scale_by_ptr,
                                const T* down_scale_by_ptr, const int64_t* skip_if,
                                const G* model_diff, const T* model, T* updated_model) {
  if (skip_if != nullptr && *skip_if != 0) {
    const int64_t n = *num_unique_ids * embedding_size;
    CUDA_1D_KERNEL_LOOP(i, n) { updated_model[i] = model[i]; }
  } else {
    if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
    if (down_scale_by_ptr != nullptr) { scale /= *down_scale_by_ptr; }
    float learning_rate_val = *learning_rate;
    const int64_t n = *num_unique_ids * embedding_size;
    CUDA_1D_KERNEL_LOOP(i, n) {
      updated_model[i] = model[i];
      SGDUpdateFunctor<T, G>()(model_diff + i, updated_model + i, scale, l1, l2, weight_decay,
                               learning_rate_val);
    }
  }
}

__device__ void GetMomentumOffset(const int32_t line_size, const int32_t embedding_size,
                                  int64_t model_diff_offset, int64_t* model_offset,
                                  int64_t* momentum_offset) {
  const int32_t row = model_diff_offset / embedding_size;
  const int32_t col = model_diff_offset - row * embedding_size;
  *model_offset = row * line_size + col;
  *momentum_offset = *model_offset + embedding_size;
}

template<typename T, typename G, typename IDX>
__global__ void MomentumUpdateKernel(const int64_t line_size, const int64_t embedding_size, T scale,
                                     float l1, float l2, float weight_decay, float beta,
                                     const IDX* num_unique_ids, const float* learning_rate,
                                     const T* scale_by_ptr, const T* down_scale_by_ptr,
                                     const int64_t* skip_if, const G* model_diff,
                                     const T* unique_values, T* updated_unique_values) {
  if (skip_if != nullptr && *skip_if != 0) {
    const int64_t n = *num_unique_ids * line_size;
    CUDA_1D_KERNEL_LOOP(i, n) { updated_unique_values[i] = unique_values[i]; }
  } else {
    if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
    if (down_scale_by_ptr != nullptr) { scale /= *down_scale_by_ptr; }
    float learning_rate_val = *learning_rate;
    const int64_t n = *num_unique_ids * embedding_size;
    CUDA_1D_KERNEL_LOOP(i, n) {
      int64_t model_offset;
      int64_t momentum_offset;
      GetMomentumOffset(line_size, embedding_size, i, &model_offset, &momentum_offset);
      updated_unique_values[model_offset] = unique_values[model_offset];
      updated_unique_values[momentum_offset] = unique_values[momentum_offset];
      MomentumUpdateFunctor<T, G>()(model_diff + i, updated_unique_values + model_offset,
                                    updated_unique_values + momentum_offset, scale, l1, l2, beta,
                                    weight_decay, learning_rate_val);
    }
  }
}

__device__ void GetAdamOffset(const int32_t line_size, const int32_t embedding_size,
                              int64_t model_diff_offset, int64_t* model_offset, int64_t* m_offset,
                              int64_t* v_offset) {
  const int32_t row = model_diff_offset / embedding_size;
  const int32_t col = model_diff_offset - row * embedding_size;
  *model_offset = row * line_size + col;
  *m_offset = *model_offset + embedding_size;
  *v_offset = *model_offset + 2 * embedding_size;
}

template<typename T, typename G, typename IDX>
__global__ void AdamUpdateKernel(const int32_t line_size, const int32_t embedding_size, T scale,
                                 float l1, float l2, float weight_decay, float beta1, float beta2,
                                 float epsilon, const float* bias_correction1_ptr,
                                 const float* bias_correction2_ptr, const IDX* num_unique_ids,
                                 const float* learning_rate, const T* scale_by_ptr,
                                 const T* down_scale_by_ptr, const int64_t* skip_if,
                                 const G* model_diff, const T* unique_values,
                                 T* updated_unique_values) {
  if (skip_if != nullptr && *skip_if != 0) {
    const int64_t n = *num_unique_ids * line_size;
    CUDA_1D_KERNEL_LOOP(i, n) {
      // The n is the unique_values elem_cnt, so not need to use GetAdamOffset.
      updated_unique_values[i] = unique_values[i];
    }
  } else {
    if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
    if (down_scale_by_ptr != nullptr) { scale /= *down_scale_by_ptr; }
    float bias_correction1_val = 1.0;
    float bias_correction2_val = 1.0;
    if (bias_correction1_ptr != nullptr) { bias_correction1_val = *bias_correction1_ptr; }
    if (bias_correction2_ptr != nullptr) { bias_correction2_val = *bias_correction2_ptr; }
    float learning_rate_val = *learning_rate;
    const int64_t n = *num_unique_ids * embedding_size;
    // The n is model_diff elem_cnt.
    CUDA_1D_KERNEL_LOOP(i, n) {
      int64_t model_offset;
      int64_t m_offset;
      int64_t v_offset;
      GetAdamOffset(line_size, embedding_size, i, &model_offset, &m_offset, &v_offset);
      updated_unique_values[model_offset] = unique_values[model_offset];
      updated_unique_values[m_offset] = unique_values[m_offset];
      updated_unique_values[v_offset] = unique_values[v_offset];
      AdamUpdateFunctor<T, G>()(model_diff + i, updated_unique_values + model_offset,
                                updated_unique_values + m_offset, updated_unique_values + v_offset,
                                nullptr, scale, l1, l2, beta1, beta2, epsilon, weight_decay, false,
                                bias_correction1_val, bias_correction2_val, learning_rate_val);
    }
  }
}

template<typename T, typename G, typename IDX>
__global__ void AdagradUpdateKernel(const int64_t line_size, const int64_t embedding_size, T scale,
                                    float l1, float l2, float weight_decay, float lr_decay,
                                    float epsilon, const IDX* num_unique_ids,
                                    const float* learning_rate, const int64_t* train_step_ptr,
                                    const T* scale_by_ptr, const T* down_scale_by_ptr,
                                    const int64_t* skip_if, const G* model_diff,
                                    const T* unique_values, T* updated_unique_values) {
  if (skip_if != nullptr && *skip_if != 0) {
    const int64_t n = *num_unique_ids * line_size;
    CUDA_1D_KERNEL_LOOP(i, n) { updated_unique_values[i] = unique_values[i]; }
  } else {
    int64_t train_step = *train_step_ptr + 1;
    if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
    if (down_scale_by_ptr != nullptr) { scale /= *down_scale_by_ptr; }
    float learning_rate_val = *learning_rate;
    learning_rate_val = learning_rate_val / (1 + (train_step - 1) * lr_decay);
    const int64_t n = *num_unique_ids * embedding_size;
    CUDA_1D_KERNEL_LOOP(i, n) {
      int64_t model_offset;
      int64_t sum_offset;
      GetMomentumOffset(line_size, embedding_size, i, &model_offset, &sum_offset);
      updated_unique_values[model_offset] = unique_values[model_offset];
      updated_unique_values[sum_offset] = unique_values[sum_offset];
      AdagradUpdateFunctor<T, G>()(model_diff + i, updated_unique_values + model_offset,
                                   updated_unique_values + sum_offset, scale, l1, l2, epsilon,
                                   weight_decay, learning_rate_val);
    }
  }
}

__device__ void GetFtrlOffset(const int32_t line_size, const int32_t embedding_size,
                              int64_t model_diff_offset, int64_t* model_offset,
                              int64_t* accumulate_offset, int64_t* z_offset) {
  const int32_t row = model_diff_offset / embedding_size;
  const int32_t col = model_diff_offset - row * embedding_size;
  *model_offset = row * line_size + col;
  *accumulate_offset = *model_offset + embedding_size;
  *z_offset = *model_offset + 2 * embedding_size;
}

template<typename T, typename G, typename IDX>
__global__ void FtrlUpdateKernel(const int32_t line_size, const int32_t embedding_size, T scale,
                                 float l1, float l2, float weight_decay, float lr_power,
                                 float lambda1, float lambda2, float beta,
                                 const IDX* num_unique_ids, const float* learning_rate,
                                 const T* down_scale_by_ptr, const int64_t* skip_if,
                                 const G* model_diff, const T* unique_values,
                                 T* updated_unique_values) {
  if (skip_if != nullptr && *skip_if != 0) {
    const int64_t n = *num_unique_ids * line_size;
    CUDA_1D_KERNEL_LOOP(i, n) { updated_unique_values[i] = unique_values[i]; }
  } else {
    if (down_scale_by_ptr != nullptr) { scale /= *down_scale_by_ptr; }
    float learning_rate_val = *learning_rate;
    const int64_t n = *num_unique_ids * embedding_size;
    CUDA_1D_KERNEL_LOOP(i, n) {
      int64_t model_offset;
      int64_t accumulate_offset;
      int64_t z_offset;
      GetFtrlOffset(line_size, embedding_size, i, &model_offset, &accumulate_offset, &z_offset);
      updated_unique_values[model_offset] = unique_values[model_offset];
      updated_unique_values[accumulate_offset] = unique_values[accumulate_offset];
      updated_unique_values[z_offset] = unique_values[z_offset];
      FtrlUpdateFunctor<T, G>()(model_diff + i, updated_unique_values + model_offset,
                                updated_unique_values + accumulate_offset,
                                updated_unique_values + z_offset, scale, l1, l2, lr_power, lambda1,
                                lambda2, beta, weight_decay, learning_rate_val);
    }
  }
}

}  // namespace

template<typename T, typename G, typename IDX>
class SgdEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  SgdEmbeddingUpdateKernel() = default;
  ~SgdEmbeddingUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    CHECK_EQ(unique_embeddings->shape().NumAxes(), 2);
    CHECK_EQ(embedding_grad->shape().NumAxes(), 2);
    const int64_t line_size = unique_embeddings->shape().At(1);
    const int64_t embedding_size = embedding_grad->shape().At(1);
    CHECK_EQ(line_size, embedding_size);
    const auto scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const T* down_scale_by_ptr = nullptr;
    if (ctx->has_input("down_scale_by_tensor", 0)) {
      const user_op::Tensor* down_scale_by_tensor =
          ctx->Tensor4ArgNameAndIndex("down_scale_by_tensor", 0);
      CHECK_EQ(down_scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(down_scale_by_tensor->shape().elem_cnt(), 1);
      down_scale_by_ptr = down_scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    // update kernel
    SGDUpdateKernel<T, G, IDX>
        <<<BlocksNum4ThreadsNum(embedding_grad->shape().elem_cnt()), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            embedding_size, scale, l1, l2, weight_decay,
            reinterpret_cast<const IDX*>(num_unique_ids->dptr()), learning_rate_ptr, scale_by_ptr,
            down_scale_by_ptr, skip_if_ptr, embedding_grad->dptr<G>(), unique_embeddings->dptr<T>(),
            updated_unique_embeddings->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define IDX_DATA_TYPE_SEQ                           \
  OF_PP_MAKE_TUPLE_SEQ(uint32_t, DataType::kUInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32)

#define REGISTER_CUDA_SGD_EMBEDDING_UPDATE_KERNEL(t_dtype_pair, g_type_pair, idx_dtype_pair)      \
  REGISTER_USER_KERNEL("sgd_embedding_update")                                                    \
      .SetCreateFn<                                                                               \
          SgdEmbeddingUpdateKernel<OF_PP_PAIR_FIRST(t_dtype_pair), OF_PP_PAIR_FIRST(g_type_pair), \
                                   OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                           \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))     \
          && (user_op::HobDataType("embedding_grad", 0) == OF_PP_PAIR_SECOND(g_type_pair))        \
          && (user_op::HobDataType("unique_embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_SGD_EMBEDDING_UPDATE_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename G, typename IDX>
class MomentumEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  MomentumEmbeddingUpdateKernel() = default;
  ~MomentumEmbeddingUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    CHECK_EQ(unique_embeddings->shape().NumAxes(), 2);
    CHECK_EQ(embedding_grad->shape().NumAxes(), 2);
    const int64_t num_keys = unique_embeddings->shape().At(0);
    const int64_t line_size = unique_embeddings->shape().At(1);
    const int64_t embedding_size = embedding_grad->shape().At(1);
    CHECK_EQ(line_size, embedding_size * 2);
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const auto beta = ctx->Attr<float>("beta");
    const auto scale = ctx->Attr<double>("scale");
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const T* down_scale_by_ptr = nullptr;
    if (ctx->has_input("down_scale_by_tensor", 0)) {
      const user_op::Tensor* down_scale_by_tensor =
          ctx->Tensor4ArgNameAndIndex("down_scale_by_tensor", 0);
      CHECK_EQ(down_scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(down_scale_by_tensor->shape().elem_cnt(), 1);
      down_scale_by_ptr = down_scale_by_tensor->dptr<T>();
    }
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    // update kernel
    MomentumUpdateKernel<T, G, IDX>
        <<<BlocksNum4ThreadsNum(embedding_grad->shape().elem_cnt()), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            line_size, embedding_size, scale, l1, l2, weight_decay, beta,
            reinterpret_cast<const IDX*>(num_unique_ids->dptr()), learning_rate_ptr, scale_by_ptr,
            down_scale_by_ptr, skip_if_ptr, embedding_grad->dptr<G>(), unique_embeddings->dptr<T>(),
            updated_unique_embeddings->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_MOMENTUM_EMBEDDING_UPDATE_KERNEL(t_dtype_pair, g_type_pair, idx_dtype_pair) \
  REGISTER_USER_KERNEL("momentum_embedding_update")                                               \
      .SetCreateFn<MomentumEmbeddingUpdateKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                  \
                                                 OF_PP_PAIR_FIRST(g_type_pair),                   \
                                                 OF_PP_PAIR_FIRST(idx_dtype_pair)>>()             \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                         \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))     \
          && (user_op::HobDataType("embedding_grad", 0) == OF_PP_PAIR_SECOND(g_type_pair))        \
          && (user_op::HobDataType("unique_embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_MOMENTUM_EMBEDDING_UPDATE_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ,
                                 IDX_DATA_TYPE_SEQ)

template<typename T, typename G, typename IDX>
class AdamEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  AdamEmbeddingUpdateKernel() = default;
  ~AdamEmbeddingUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    CHECK_EQ(unique_embeddings->shape().NumAxes(), 2);
    CHECK_EQ(embedding_grad->shape().NumAxes(), 2);
    const int64_t num_keys = unique_embeddings->shape().At(0);
    const int64_t line_size = unique_embeddings->shape().At(1);
    const int64_t embedding_size = embedding_grad->shape().At(1);
    CHECK_EQ(line_size, embedding_size * 3);

    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const auto beta1 = ctx->Attr<float>("beta1");
    const auto beta2 = ctx->Attr<float>("beta2");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const bool do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    const auto scale = ctx->Attr<double>("scale");
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const T* down_scale_by_ptr = nullptr;
    if (ctx->has_input("down_scale_by_tensor", 0)) {
      const user_op::Tensor* down_scale_by_tensor =
          ctx->Tensor4ArgNameAndIndex("down_scale_by_tensor", 0);
      CHECK_EQ(down_scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(down_scale_by_tensor->shape().elem_cnt(), 1);
      down_scale_by_ptr = down_scale_by_tensor->dptr<T>();
    }
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    const float* bias_correction1_ptr = nullptr;
    if (ctx->has_input("bias_correction1", 0)) {
      bias_correction1_ptr = ctx->Tensor4ArgNameAndIndex("bias_correction1", 0)->dptr<float>();
    }
    const float* bias_correction2_ptr = nullptr;
    if (ctx->has_input("bias_correction2", 0)) {
      bias_correction2_ptr = ctx->Tensor4ArgNameAndIndex("bias_correction2", 0)->dptr<float>();
    }
    // update kernel
    AdamUpdateKernel<T, G, IDX>
        <<<BlocksNum4ThreadsNum(embedding_grad->shape().elem_cnt()), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            line_size, embedding_size, static_cast<T>(scale), l1, l2, weight_decay, beta1, beta2,
            epsilon, bias_correction1_ptr, bias_correction2_ptr,
            reinterpret_cast<const IDX*>(num_unique_ids->dptr()), learning_rate_ptr, scale_by_ptr,
            down_scale_by_ptr, skip_if_ptr, embedding_grad->dptr<G>(), unique_embeddings->dptr<T>(),
            updated_unique_embeddings->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ADAM_EMBEDDING_UPDATE_KERNEL(t_dtype_pair, g_type_pair, idx_dtype_pair)      \
  REGISTER_USER_KERNEL("adam_embedding_update")                                                    \
      .SetCreateFn<                                                                                \
          AdamEmbeddingUpdateKernel<OF_PP_PAIR_FIRST(t_dtype_pair), OF_PP_PAIR_FIRST(g_type_pair), \
                                    OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                           \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                          \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))      \
          && (user_op::HobDataType("embedding_grad", 0) == OF_PP_PAIR_SECOND(g_type_pair))         \
          && (user_op::HobDataType("unique_embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ADAM_EMBEDDING_UPDATE_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

template<typename T, typename G, typename IDX>
class AdagradEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  AdagradEmbeddingUpdateKernel() = default;
  ~AdagradEmbeddingUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    CHECK_EQ(unique_embeddings->shape().NumAxes(), 2);
    CHECK_EQ(embedding_grad->shape().NumAxes(), 2);
    const int64_t num_keys = unique_embeddings->shape().At(0);
    const int64_t line_size = unique_embeddings->shape().At(1);
    const int64_t embedding_size = embedding_grad->shape().At(1);
    CHECK_EQ(line_size, embedding_size * 2);

    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const auto lr_decay = ctx->Attr<float>("lr_decay");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto scale = ctx->Attr<double>("scale");
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const T* down_scale_by_ptr = nullptr;
    if (ctx->has_input("down_scale_by_tensor", 0)) {
      const user_op::Tensor* down_scale_by_tensor =
          ctx->Tensor4ArgNameAndIndex("down_scale_by_tensor", 0);
      CHECK_EQ(down_scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(down_scale_by_tensor->shape().elem_cnt(), 1);
      down_scale_by_ptr = down_scale_by_tensor->dptr<T>();
    }
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const int64_t* train_step_ptr = ctx->Tensor4ArgNameAndIndex("train_step", 0)->dptr<int64_t>();
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    // update kernel
    AdagradUpdateKernel<T, G, IDX>
        <<<BlocksNum4ThreadsNum(embedding_grad->shape().elem_cnt()), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            line_size, embedding_size, static_cast<T>(scale), l1, l2, weight_decay, lr_decay,
            epsilon, reinterpret_cast<const IDX*>(num_unique_ids->dptr()), learning_rate_ptr,
            train_step_ptr, scale_by_ptr, down_scale_by_ptr, skip_if_ptr, embedding_grad->dptr<G>(),
            unique_embeddings->dptr<T>(), updated_unique_embeddings->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ADAGRAD_EMBEDDING_UPDATE_KERNEL(t_dtype_pair, g_type_pair, idx_dtype_pair) \
  REGISTER_USER_KERNEL("adagrad_embedding_update")                                               \
      .SetCreateFn<AdagradEmbeddingUpdateKernel<OF_PP_PAIR_FIRST(t_dtype_pair),                  \
                                                OF_PP_PAIR_FIRST(g_type_pair),                   \
                                                OF_PP_PAIR_FIRST(idx_dtype_pair)>>()             \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                        \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))    \
          && (user_op::HobDataType("embedding_grad", 0) == OF_PP_PAIR_SECOND(g_type_pair))       \
          && (user_op::HobDataType("unique_embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_ADAGRAD_EMBEDDING_UPDATE_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ,
                                 IDX_DATA_TYPE_SEQ)

template<typename T, typename G, typename IDX>
class FtrlEmbeddingUpdateKernel final : public user_op::OpKernel {
 public:
  FtrlEmbeddingUpdateKernel() = default;
  ~FtrlEmbeddingUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* num_unique_ids = ctx->Tensor4ArgNameAndIndex("num_unique_ids", 0);
    const user_op::Tensor* unique_embeddings = ctx->Tensor4ArgNameAndIndex("unique_embeddings", 0);
    const user_op::Tensor* embedding_grad = ctx->Tensor4ArgNameAndIndex("embedding_grad", 0);
    user_op::Tensor* updated_unique_embeddings =
        ctx->Tensor4ArgNameAndIndex("updated_unique_embeddings", 0);
    CHECK_EQ(unique_embeddings->shape().NumAxes(), 2)
        << "The NumAxes of unique_embedding should be equal to 2. ";
    CHECK_EQ(embedding_grad->shape().NumAxes(), 2)
        << "The NumAxes of embedding_grad should be equal to 2. ";
    const int64_t num_keys = unique_embeddings->shape().At(0);
    const int64_t line_size = unique_embeddings->shape().At(1);
    const int64_t embedding_size = embedding_grad->shape().At(1);
    CHECK_EQ(line_size, embedding_size * 3)
        << "The line_size should be equal to 3 x embedding_size. ";
    const float l1 = 0.0;
    const float l2 = 0.0;
    const float weight_decay = ctx->Attr<float>("weight_decay");
    // TODO(zhengzekang): Undefined behavior for ftrl optimizer with weight_decay in `abs(new_z_val)
    // < lambda1` condition.
    CHECK_EQ(weight_decay, static_cast<float>(0.0))
        << "Currently not support for setting weight decay. ";
    const float lr_power = ctx->Attr<float>("lr_power");
    const float lambda1 = ctx->Attr<float>("lambda1");
    const float lambda2 = ctx->Attr<float>("lambda2");
    const float beta = ctx->Attr<float>("beta");
    const double scale = ctx->Attr<double>("scale");
    const T* down_scale_by_ptr = nullptr;
    if (ctx->has_input("down_scale_by_tensor", 0)) {
      const user_op::Tensor* down_scale_by_tensor =
          ctx->Tensor4ArgNameAndIndex("down_scale_by_tensor", 0);
      CHECK_EQ(down_scale_by_tensor->data_type(), unique_embeddings->data_type());
      CHECK_EQ(down_scale_by_tensor->shape().elem_cnt(), 1);
      down_scale_by_ptr = down_scale_by_tensor->dptr<T>();
    }
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const float* learning_rate_ptr = learning_rate->dptr<float>();
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    // update kernel
    FtrlUpdateKernel<T, G, IDX>
        <<<BlocksNum4ThreadsNum(embedding_grad->shape().elem_cnt()), kCudaThreadsNumPerBlock, 0,
           ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            line_size, embedding_size, static_cast<T>(scale), l1, l2, weight_decay, lr_power,
            lambda1, lambda2, beta, reinterpret_cast<const IDX*>(num_unique_ids->dptr()),
            learning_rate_ptr, down_scale_by_ptr, skip_if_ptr, embedding_grad->dptr<G>(),
            unique_embeddings->dptr<T>(), updated_unique_embeddings->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_CUDA_FTRL_EMBEDDING_UPDATE_KERNEL(t_dtype_pair, g_type_pair, idx_dtype_pair)      \
  REGISTER_USER_KERNEL("ftrl_embedding_update")                                                    \
      .SetCreateFn<                                                                                \
          FtrlEmbeddingUpdateKernel<OF_PP_PAIR_FIRST(t_dtype_pair), OF_PP_PAIR_FIRST(g_type_pair), \
                                    OF_PP_PAIR_FIRST(idx_dtype_pair)>>()                           \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceType() == DeviceType::kCUDA)                                          \
          && (user_op::HobDataType("num_unique_ids", 0) == OF_PP_PAIR_SECOND(idx_dtype_pair))      \
          && (user_op::HobDataType("embedding_grad", 0) == OF_PP_PAIR_SECOND(g_type_pair))         \
          && (user_op::HobDataType("unique_embeddings", 0) == OF_PP_PAIR_SECOND(t_dtype_pair)));
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CUDA_FTRL_EMBEDDING_UPDATE_KERNEL, FLOATING_DATA_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, IDX_DATA_TYPE_SEQ)

}  // namespace oneflow
