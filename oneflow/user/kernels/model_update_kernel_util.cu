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
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/user/kernels/model_update_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename G>
__global__ void SGDUpdateGpu(int64_t n, T scale, float l1, float l2, float weight_decay,
                             const float* learning_rate, const T* scale_by_ptr, const G* model_diff,
                             T* model) {
  const T lr = *learning_rate;
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  CUDA_1D_KERNEL_LOOP(i, n) {
    SGDUpdateFunctor<T, G>()(model_diff + i, model + i, scale, l1, l2, weight_decay, lr);
  }
}

template<typename T, typename K, typename IDX>
__global__ void IndexedSlicesSGDUpdateGpu(const IDX data_elem_cnt, const K* indices,
                                          const T* values, const float* learning_rate,
                                          const IDX num_features, const IDX feature_size, T* model,
                                          const IDX feature_id_offset) {
  const T minus_lr = -*learning_rate;
  CUDA_1D_KERNEL_LOOP_T(IDX, i, data_elem_cnt) {
    const T val = values[i];
    if (val != static_cast<T>(0)) {
      const IDX indices_idx = i / feature_size;
      const IDX inner_idx = i - indices_idx * feature_size;
      const IDX feature_id = indices[indices_idx];
      assert(feature_id >= 0);
      const IDX local_feature_id = feature_id - feature_id_offset;
      if (local_feature_id >= 0 && local_feature_id < num_features) {
        const IDX update_offset = local_feature_id * feature_size + inner_idx;
        gpu_atomic_add(model + update_offset, val * minus_lr);
      }
    }
  }
}

}  // namespace

template<typename T, typename G>
struct SGDUpdateKernelUtil<DeviceType::kGPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float weight_decay,
                     const float* learning_rate, const T* scale_by_ptr, const G* model_diff,
                     T* model);
};

template<typename T, typename G>
void SGDUpdateKernelUtil<DeviceType::kGPU, T, G>::Update(DeviceCtx* ctx, int64_t n, T scale,
                                                         float l1, float l2, float weight_decay,
                                                         const float* learning_rate,
                                                         const T* scale_by_ptr, const G* model_diff,
                                                         T* model) {
  SGDUpdateGpu<T, G><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, scale, l1, l2, weight_decay, learning_rate, scale_by_ptr, model_diff, model);
}

template<typename T>
struct SGDUpdateKernelUtil<DeviceType::kGPU, T, float16> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float weight_decay,
                     const float* learning_rate, const T* scale_by_ptr, const float16* model_diff,
                     T* model);
};

template<typename T>
void SGDUpdateKernelUtil<DeviceType::kGPU, T, float16>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float weight_decay,
    const float* learning_rate, const T* scale_by_ptr, const float16* model_diff, T* model) {
  SGDUpdateKernelUtil<DeviceType::kGPU, T, half>::Update(
      ctx, n, scale, l1, l2, weight_decay, learning_rate, scale_by_ptr,
      reinterpret_cast<const half*>(model_diff), model);
}

template struct SGDUpdateKernelUtil<DeviceType::kGPU, double, double>;
template struct SGDUpdateKernelUtil<DeviceType::kGPU, float, float>;
template struct SGDUpdateKernelUtil<DeviceType::kGPU, float, float16>;

template<typename T, typename K>
struct IndexedSlicesSGDUpdateKernelUtil<DeviceType::kGPU, T, K> {
  static void Update(DeviceCtx* ctx, int64_t num_indices, int64_t num_features,
                     int64_t feature_size, int64_t feature_id_offset, const float* learning_rate,
                     const K* indices, const T* values, T* model);
};

template<typename T, typename K>
void IndexedSlicesSGDUpdateKernelUtil<DeviceType::kGPU, T, K>::Update(
    DeviceCtx* ctx, int64_t num_indices, int64_t num_features, int64_t feature_size,
    int64_t feature_id_offset, const float* learning_rate, const K* indices, const T* values,
    T* model) {
  const int64_t values_elem_cnt = num_indices * feature_size;
  IndexedSlicesSGDUpdateGpu<T, K, int64_t>
      <<<BlocksNum4ThreadsNum(values_elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          values_elem_cnt, indices, values, learning_rate, num_features, feature_size, model,
          feature_id_offset);
}

#define INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_GPU(in_type_pair, index_type_pair) \
  template struct IndexedSlicesSGDUpdateKernelUtil<                                       \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_GPU

namespace {

template<typename T, typename G>
__global__ void MomentumUpdateGpu(int64_t n, T scale, float l1, float l2, float beta,
                                  float weight_decay, const float* learning_rate,
                                  const T* scale_by_ptr, const G* model_diff, T* model,
                                  T* momentum) {
  const T lr = *learning_rate;
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  CUDA_1D_KERNEL_LOOP(i, n) {
    MomentumUpdateFunctor<T, G>()(model_diff + i, model + i, momentum + i, scale, l1, l2, beta,
                                  weight_decay, lr);
  }
}

template<typename T, typename K, typename IDX>
__global__ void IndexedSlicesMomentumUpdateGpu(T beta, int64_t feature_size, int64_t lower_bound,
                                               int64_t upper_bound, const IDX* num_unique_instance,
                                               const float* learning_rate, const K* indices,
                                               const T* values, T* model, T* momentum) {
  const int64_t n = *num_unique_instance * feature_size;
  const T lr = *learning_rate;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const IDX indices_idx = i / feature_size;
    const IDX inner_idx = i - indices_idx * feature_size;
    const IDX instance_id = indices[indices_idx];
    if (instance_id >= lower_bound && instance_id < upper_bound) {
      const IDX model_idx = (instance_id - lower_bound) * feature_size + inner_idx;
      MomentumUpdateFunctor<T, T>()(values + i, model + model_idx, momentum + model_idx,
                                    static_cast<T>(1), 0.0, 0.0, beta, 0.0, lr);
    }
  }
}

}  // namespace

template<typename T, typename G>
struct MomentumUpdateKernelUtil<DeviceType::kGPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta,
                     float weight_decay, const float* learning_rate, const T* scale_by_ptr,
                     const G* model_diff, T* model, T* momentum);
};

template<typename T, typename G>
void MomentumUpdateKernelUtil<DeviceType::kGPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta, float weight_decay,
    const float* learning_rate, const T* scale_by_ptr, const G* model_diff, T* model, T* momentum) {
  MomentumUpdateGpu<T, G>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, scale, l1, l2, beta, weight_decay, learning_rate, scale_by_ptr, model_diff, model,
          momentum);
}

template<typename T>
struct MomentumUpdateKernelUtil<DeviceType::kGPU, T, float16> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta,
                     float weight_decay, const float* learning_rate, const T* scale_by_ptr,
                     const float16* model_diff, T* model, T* momentum);
};

template<typename T>
void MomentumUpdateKernelUtil<DeviceType::kGPU, T, float16>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta, float weight_decay,
    const float* learning_rate, const T* scale_by_ptr, const float16* model_diff, T* model,
    T* momentum) {
  MomentumUpdateKernelUtil<DeviceType::kGPU, T, half>::Update(
      ctx, n, scale, l1, l2, beta, weight_decay, learning_rate, scale_by_ptr,
      reinterpret_cast<const half*>(model_diff), model, momentum);
}

template struct MomentumUpdateKernelUtil<DeviceType::kGPU, double, double>;
template struct MomentumUpdateKernelUtil<DeviceType::kGPU, float, float>;
template struct MomentumUpdateKernelUtil<DeviceType::kGPU, float, float16>;

template<typename T, typename K, typename IDX>
struct IndexedSlicesMomentumMdUpdateKernelUtil<DeviceType::kGPU, T, K, IDX> {
  static void Update(DeviceCtx* ctx, T beta, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const float* learning_rate, const K* indices, const T* values, T* model,
                     T* momentum);
};

template<typename T, typename K, typename IDX>
void IndexedSlicesMomentumMdUpdateKernelUtil<DeviceType::kGPU, T, K, IDX>::Update(
    DeviceCtx* ctx, T beta, int64_t num_instance, int64_t feature_size, int64_t lower_bound,
    int64_t upper_bound, const IDX* num_unique_instance, const float* learning_rate,
    const K* indices, const T* values, T* model, T* momentum) {
  IndexedSlicesMomentumUpdateGpu<T, K>
      <<<BlocksNum4ThreadsNum(num_instance * feature_size), kCudaThreadsNumPerBlock, 0,
         ctx->cuda_stream()>>>(beta, feature_size, lower_bound, upper_bound, num_unique_instance,
                               learning_rate, indices, values, model, momentum);
}

#define INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_GPU(                 \
    val_type_pair, key_type_pair, idx_type_pair)                                          \
  template struct IndexedSlicesMomentumMdUpdateKernelUtil<                                \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(val_type_pair), OF_PP_PAIR_FIRST(key_type_pair), \
      OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_GPU

namespace {

__global__ void AdamBiasCorrectionLearningRateGpu(float beta1, float beta2,
                                                  const float* learning_rate,
                                                  const int64_t* train_step, float* out) {
  const auto exponent = static_cast<double>(*train_step + 1);
  const float beta1_power = static_cast<float>(pow(beta1, exponent));
  const float beta2_power = static_cast<float>(pow(beta2, exponent));
  *out = *learning_rate * sqrt(1 - beta2_power) / (1 - beta1_power);
}

template<typename T, typename G>
__global__ void AdamUpdateGpu(int64_t n, T scale, float l1, float l2, float beta1, float beta2,
                              float epsilon, float weight_decay, const float* learning_rate,
                              const T* scale_by_ptr, const G* model_diff, T* model, T* m, T* v) {
  const float lr = *learning_rate;
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  CUDA_1D_KERNEL_LOOP(i, n) {
    AdamUpdateFunctor<T, G>()(model_diff + i, model + i, m + i, v + i, scale, l1, l2, beta1, beta2,
                              epsilon, weight_decay, lr);
  }
}

template<typename T>
__global__ void AdamUpdateBetaTGpu(const T beta1, const T beta2, T* beta1_t, T* beta2_t) {
  *beta1_t *= beta1;
  *beta2_t *= beta2;
}

template<typename T, typename K, typename IDX>
__global__ void IndexedSlicesAdamUpdateGpu(float beta1, float beta2, float epsilon,
                                           int64_t feature_size, int64_t lower_bound,
                                           int64_t upper_bound, const IDX* num_unique_instance,
                                           const float* learning_rate, const K* indices,
                                           const T* values, T* model, T* m, T* v) {
  const float lr = *learning_rate;
  const int64_t n = *num_unique_instance * feature_size;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const IDX indices_idx = i / feature_size;
    const IDX inner_idx = i - indices_idx * feature_size;
    const IDX instance_id = indices[indices_idx];
    if (instance_id >= lower_bound && instance_id < upper_bound) {
      const IDX model_idx = (instance_id - lower_bound) * feature_size + inner_idx;
      AdamUpdateFunctor<T, T>()(values + i, model + model_idx, m + model_idx, v + model_idx,
                                static_cast<T>(1), 0, 0, beta1, beta2, epsilon, 0, lr);
    }
  }
}

template<typename T, typename G>
__global__ void LambGradGpu(int64_t n, T scale, float l1, float l2, float beta1, float beta2,
                            float epsilon, const T* beta1_t, const T* beta2_t,
                            const T* scale_by_ptr, const G* model_diff, T* adam_diff, T* model,
                            T* m, T* v) {
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  CUDA_1D_KERNEL_LOOP(i, n) {
    LambGradFunctor<T, G>()(beta1_t, beta2_t, model_diff + i, adam_diff + i, model + i, m + i,
                            v + i, scale, l1, l2, beta1, beta2, epsilon);
  }
}

template<typename T>
__global__ void LambUpdateGpu(int64_t n, float weight_decay, const float* learning_rate,
                              const T* w_norm, const T* g_norm, const T* beta1_t, const T* beta2_t,
                              const T* adam_diff, T* model) {
  const float lr = LambLRFunctor<T>()(*learning_rate, w_norm, g_norm);
  CUDA_1D_KERNEL_LOOP(i, n) { LambUpdateFunctor<T>()(lr, weight_decay, adam_diff + i, model + i); }
}

}  // namespace

template<typename T, typename G>
struct AdamUpdateKernelUtil<DeviceType::kGPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, const float* learning_rate,
                     const T* scale_by_ptr, const G* model_diff, T* model, T* m, T* v);
};

template<typename T, typename G>
void AdamUpdateKernelUtil<DeviceType::kGPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta1, float beta2, float epsilon,
    float weight_decay, const float* learning_rate, const T* scale_by_ptr, const G* model_diff,
    T* model, T* m, T* v) {
  AdamUpdateGpu<T, G><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, scale, l1, l2, beta1, beta2, epsilon, weight_decay, learning_rate, scale_by_ptr,
      model_diff, model, m, v);
}

template<typename T>
struct AdamUpdateKernelUtil<DeviceType::kGPU, T, float16> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, const float* learning_rate,
                     const T* scale_by_ptr, const float16* model_diff, T* model, T* m, T* v);
};

template<typename T>
void AdamUpdateKernelUtil<DeviceType::kGPU, T, float16>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta1, float beta2, float epsilon,
    float weight_decay, const float* learning_rate, const T* scale_by_ptr,
    const float16* model_diff, T* model, T* m, T* v) {
  AdamUpdateKernelUtil<DeviceType::kGPU, T, half>::Update(
      ctx, n, scale, l1, l2, beta1, beta2, epsilon, weight_decay, learning_rate, scale_by_ptr,
      reinterpret_cast<const half*>(model_diff), model, m, v);
}

template struct AdamUpdateKernelUtil<DeviceType::kGPU, float, float>;
template struct AdamUpdateKernelUtil<DeviceType::kGPU, double, double>;
template struct AdamUpdateKernelUtil<DeviceType::kGPU, float, float16>;

template<typename T, typename G>
struct LambUpdateKernelUtil<DeviceType::kGPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, const float* learning_rate,
                     const T* scale_by_ptr, const G* model_diff, T* adam_diff, T* model, T* m, T* v,
                     T* norm_buffer, T* beta1_t, T* beta2_t);
};

template<typename T, typename G>
void LambUpdateKernelUtil<DeviceType::kGPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1, float beta2,
    float epsilon, float weight_decay, const float* learning_rate, const T* scale_by_ptr,
    const G* model_diff, T* adam_diff, T* model, T* m, T* v, T* norm_buffer, T* beta1_t,
    T* beta2_t) {
  AdamUpdateBetaTGpu<T><<<1, 1, 0, ctx->cuda_stream()>>>(beta1, beta2, beta1_t, beta2_t);
  LambGradGpu<T, G><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, scale, l1, l2, beta1, beta2, epsilon, beta1_t, beta2_t, scale_by_ptr, model_diff,
      adam_diff, model, m, v);
  T* w_norm = norm_buffer;
  T* g_norm = norm_buffer + 1;
  KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, model, 1, model, 1, w_norm);
  KernelUtil<DeviceType::kGPU, T>::Dot(ctx, n, adam_diff, 1, adam_diff, 1, g_norm);
  KernelUtil<DeviceType::kGPU, T>::Sqrt(ctx, 2, norm_buffer, norm_buffer);
  LambUpdateGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, weight_decay, learning_rate, w_norm, g_norm, beta1_t, beta2_t, adam_diff, model);
}

template<typename T>
struct LambUpdateKernelUtil<DeviceType::kGPU, T, float16> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, const float* learning_rate,
                     const T* scale_by_ptr, const float16* model_diff, T* adam_diff, T* model, T* m,
                     T* v, T* norm_buffer, T* beta1_t, T* beta2_t);
};

template<typename T>
void LambUpdateKernelUtil<DeviceType::kGPU, T, float16>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1, float beta2,
    float epsilon, float weight_decay, const float* learning_rate, const T* scale_by_ptr,
    const float16* model_diff, T* adam_diff, T* model, T* m, T* v, T* norm_buffer, T* beta1_t,
    T* beta2_t) {
  LambUpdateKernelUtil<DeviceType::kGPU, T, half>::Update(
      ctx, n, scale, l1, l2, beta1, beta2, epsilon, weight_decay, learning_rate, scale_by_ptr,
      reinterpret_cast<const half*>(model_diff), adam_diff, model, m, v, norm_buffer, beta1_t,
      beta2_t);
}

template struct LambUpdateKernelUtil<DeviceType::kGPU, float, float>;
template struct LambUpdateKernelUtil<DeviceType::kGPU, double, double>;
template struct LambUpdateKernelUtil<DeviceType::kGPU, float, float16>;

template<typename T, typename K, typename IDX>
struct IndexedSlicesAdamMdUpdateKernelUtil<DeviceType::kGPU, T, K, IDX> {
  static void Update(DeviceCtx* ctx, float beta1, float beta2, float epsilon, int64_t num_instance,
                     int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                     const IDX* num_unique_instance, const float* learning_rate, const K* indices,
                     const T* values, T* model, T* m, T* v);
};

template<typename T, typename K, typename IDX>
void IndexedSlicesAdamMdUpdateKernelUtil<DeviceType::kGPU, T, K, IDX>::Update(
    DeviceCtx* ctx, float beta1, float beta2, float epsilon, int64_t num_instance,
    int64_t feature_size, int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
    const float* learning_rate, const K* indices, const T* values, T* model, T* m, T* v) {
  IndexedSlicesAdamUpdateGpu<T, K>
      <<<BlocksNum4ThreadsNum(num_instance * feature_size), kCudaThreadsNumPerBlock, 0,
         ctx->cuda_stream()>>>(beta1, beta2, epsilon, feature_size, lower_bound, upper_bound,
                               num_unique_instance, learning_rate, indices, values, model, m, v);
}

#define INSTANTIATE_INDEXED_SLICES_ADAM_MODEL_UPDATE_KERNEL_UTIL_GPU(val_type_pair, key_type_pair, \
                                                                     idx_type_pair)                \
  template struct IndexedSlicesAdamMdUpdateKernelUtil<                                             \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(val_type_pair), OF_PP_PAIR_FIRST(key_type_pair),          \
      OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_ADAM_MODEL_UPDATE_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, INT_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_ADAM_MODEL_UPDATE_KERNEL_UTIL_GPU

template<>
struct AdamBiasCorrectionLearningRateKernelUtil<DeviceType::kGPU> {
  static void AdamBiasCorrectionLearningRate(DeviceCtx* ctx, float beta1, float beta2,
                                             const float* learning_rate, const int64_t* train_step,
                                             float* out);
};

void AdamBiasCorrectionLearningRateKernelUtil<DeviceType::kGPU>::AdamBiasCorrectionLearningRate(
    DeviceCtx* ctx, float beta1, float beta2, const float* learning_rate, const int64_t* train_step,
    float* out) {
  AdamBiasCorrectionLearningRateGpu<<<1, 1, 0, ctx->cuda_stream()>>>(beta1, beta2, learning_rate,
                                                                     train_step, out);
}

}  // namespace oneflow
