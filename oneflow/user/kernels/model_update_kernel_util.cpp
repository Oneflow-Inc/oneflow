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
#include "oneflow/user/kernels/model_update_kernel_util.h"

namespace oneflow {

template<typename T, typename G>
struct SGDUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float weight_decay,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     const G* model_diff, T* model);
};

template<typename T, typename G>
void SGDUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(DeviceCtx* ctx, int64_t n, T scale,
                                                         float l1, float l2, float weight_decay,
                                                         const float* learning_rate,
                                                         const T* scale_by_ptr,
                                                         const int64_t* skip_if,
                                                         const G* model_diff, T* model) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  const T lr = *learning_rate;
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  for (int64_t i = 0; i != n; ++i) {
    SGDUpdateFunctor<T, G>()(model_diff + i, model + i, scale, l1, l2, weight_decay, lr);
  }
}

template struct SGDUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct SGDUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename K>
struct IndexedSlicesSGDUpdateKernelUtil<DeviceType::kCPU, T, K> {
  static void Update(DeviceCtx* ctx, int64_t num_indices, int64_t num_features,
                     int64_t feature_size, int64_t feature_id_offset, const float* learning_rate,
                     const K* indices, const T* values, T* model);
};

template<typename T, typename K>
void IndexedSlicesSGDUpdateKernelUtil<DeviceType::kCPU, T, K>::Update(
    DeviceCtx* ctx, int64_t num_indices, int64_t num_features, int64_t feature_size,
    int64_t feature_id_offset, const float* learning_rate, const K* indices, const T* values,
    T* model) {
  FOR_RANGE(int64_t, i, 0, num_indices) {
    const K feature_id = indices[i];
    CHECK_GE(feature_id, 0);
    const K local_feature_id = feature_id - feature_id_offset;
    if (local_feature_id >= 0 && local_feature_id < num_features) {
      const T* from = values + i * feature_size;
      T* to = model + local_feature_id * feature_size;
      for (int64_t j = 0; j < feature_size; ++j) { to[j] -= from[j] * (*learning_rate); }
    }
  }
}

#define INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_CPU(in_type_pair, index_type_pair) \
  template struct IndexedSlicesSGDUpdateKernelUtil<                                       \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(in_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_CPU

template<typename T, typename G>
struct MomentumUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta,
                     float weight_decay, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* momentum);
};

template<typename T, typename G>
void MomentumUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta, float weight_decay,
    const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
    T* model, T* momentum) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  const T lr = *learning_rate;
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  for (int64_t i = 0; i != n; ++i) {
    MomentumUpdateFunctor<T, G>()(model_diff + i, model + i, momentum + i, scale, l1, l2, beta,
                                  weight_decay, lr);
  }
}

template struct MomentumUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct MomentumUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename K, typename IDX>
struct IndexedSlicesMomentumMdUpdateKernelUtil<DeviceType::kCPU, T, K, IDX> {
  static void Update(DeviceCtx* ctx, T beta, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const float* learning_rate, const K* indices, const T* values, T* model,
                     T* momentum);
};

template<typename T, typename K, typename IDX>
void IndexedSlicesMomentumMdUpdateKernelUtil<DeviceType::kCPU, T, K, IDX>::Update(
    DeviceCtx* ctx, T beta, int64_t num_instance, int64_t feature_size, int64_t lower_bound,
    int64_t upper_bound, const IDX* num_unique_instance, const float* learning_rate,
    const K* indices, const T* values, T* model, T* momentum) {
  const int64_t n = *num_unique_instance * feature_size;
  const T lr = *learning_rate;
  for (int64_t i = 0; i != n; ++i) {
    const IDX indices_idx = i / feature_size;
    const IDX inner_idx = i - indices_idx * feature_size;
    const IDX instance_id = indices[indices_idx];
    if (instance_id >= lower_bound && instance_id < upper_bound) {
      const IDX model_idx = (instance_id - lower_bound) * feature_size + inner_idx;
      MomentumUpdateFunctor<T, T>()(values + i, model + model_idx, momentum + model_idx, 1.0, 0.0,
                                    0.0, beta, 0.0, lr);
    }
  }
}

#define INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_CPU(                 \
    val_type_pair, key_type_pair, idx_type_pair)                                          \
  template struct IndexedSlicesMomentumMdUpdateKernelUtil<                                \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(val_type_pair), OF_PP_PAIR_FIRST(key_type_pair), \
      OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_MOMENTUM_MODEL_UPDATE_KERNEL_UTIL_CPU

template<typename T, typename G>
struct AdamUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     T* m, T* v);
};

template<typename T, typename G>
void AdamUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta1, float beta2, float epsilon,
    float weight_decay, const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
    const G* model_diff, T* model, T* m, T* v) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  const float lr = *learning_rate;
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  FOR_RANGE(int64_t, i, 0, n) {
    AdamUpdateFunctor<T, G>()(model_diff + i, model + i, m + i, v + i, scale, l1, l2, beta1, beta2,
                              epsilon, weight_decay, lr);
  }
}

template struct AdamUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct AdamUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename K, typename IDX>
struct IndexedSlicesAdamMdUpdateKernelUtil<DeviceType::kCPU, T, K, IDX> {
  static void Update(DeviceCtx* ctx, float beta1, float beta2, float epsilon, int64_t num_instance,
                     int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                     const IDX* num_unique_instance, const float* learning_rate, const K* indices,
                     const T* values, T* model, T* m, T* v) {
    const float lr = *learning_rate;
    const int64_t n = *num_unique_instance * feature_size;
    FOR_RANGE(int64_t, i, 0, n) {
      const IDX indices_idx = i / feature_size;
      const IDX inner_idx = i - indices_idx * feature_size;
      const IDX instance_id = indices[indices_idx];
      if (instance_id >= lower_bound && instance_id < upper_bound) {
        const IDX model_idx = (instance_id - lower_bound) * feature_size + inner_idx;
        AdamUpdateFunctor<T, T>()(values + i, model + model_idx, m + model_idx, v + model_idx, 1, 0,
                                  0, beta1, beta2, epsilon, 0, lr);
      }
    }
  }
};

#define INSTANTIATE_INDEXED_SLICES_ADAM_MODEL_UPDATE_KERNEL_UTIL_CPU(val_type_pair, key_type_pair, \
                                                                     idx_type_pair)                \
  template struct IndexedSlicesAdamMdUpdateKernelUtil<                                             \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(val_type_pair), OF_PP_PAIR_FIRST(key_type_pair),          \
      OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_INDEXED_SLICES_ADAM_MODEL_UPDATE_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_INDEXED_SLICES_ADAM_MODEL_UPDATE_KERNEL_UTIL_CPU

template<typename T, typename G>
struct LambUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
                     T* adam_diff, T* model, T* m, T* v, T* norm_buffer, T* beta1_t, T* beta2_t);
};

template<typename T, typename G>
void LambUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1, float beta2,
    float epsilon, float weight_decay, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, const G* model_diff, T* adam_diff, T* model, T* m, T* v, T* norm_buffer,
    T* beta1_t, T* beta2_t) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  *beta1_t *= beta1;
  *beta2_t *= beta2;
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  FOR_RANGE(int64_t, i, 0, n) {
    LambGradFunctor<T, G>()(beta1_t, beta2_t, model_diff + i, adam_diff + i, model + i, m + i,
                            v + i, scale, l1, l2, beta1, beta2, epsilon);
  }
  T* w_norm = norm_buffer;
  T* g_norm = norm_buffer + 1;
  KernelUtil<DeviceType::kCPU, T>::Dot(ctx, n, model, 1, model, 1, w_norm);
  KernelUtil<DeviceType::kCPU, T>::Dot(ctx, n, adam_diff, 1, adam_diff, 1, g_norm);
  KernelUtil<DeviceType::kCPU, T>::Sqrt(ctx, 2, norm_buffer, norm_buffer);
  const float lr = LambLRFunctor<T>()(*learning_rate, w_norm, g_norm);
  FOR_RANGE(int64_t, i, 0, n) {
    LambUpdateFunctor<T>()(lr, weight_decay, adam_diff + i, model + i);
  }
}

template struct LambUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct LambUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<>
struct AdamBiasCorrectionLearningRateKernelUtil<DeviceType::kCPU> {
  static void AdamBiasCorrectionLearningRate(DeviceCtx* ctx, float beta1, float beta2,
                                             const float* learning_rate, const int64_t* train_step,
                                             float* out);
};

void AdamBiasCorrectionLearningRateKernelUtil<DeviceType::kCPU>::AdamBiasCorrectionLearningRate(
    DeviceCtx* ctx, float beta1, float beta2, const float* learning_rate, const int64_t* train_step,
    float* out) {
  const auto exponent = static_cast<double>(*train_step + 1);
  const float beta1_power = static_cast<float>(std::pow(beta1, exponent));
  const float beta2_power = static_cast<float>(std::pow(beta2, exponent));
  *out = *learning_rate * sqrt(1 - beta2_power) / (1 - beta1_power);
}

template<typename T, typename G>
struct RmsPropUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, bool centered,
                     float epsilon, float weight_decay, float decay_rate,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     const G* model_diff, T* model, T* mean_square, T* mean_gradient);
};

template<typename T, typename G>
void RmsPropUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, bool centered, float epsilon,
    float weight_decay, float decay_rate, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, const G* model_diff, T* model, T* mean_square, T* mean_gradient) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  if (centered) {
    FOR_RANGE(int64_t, i, 0, n) {
      RmsPropUpdateFunctor<T, G, true>()(model_diff + i, model + i, n, scale, l1, l2,
                                         mean_square + i, mean_gradient + i, epsilon, weight_decay,
                                         decay_rate, *learning_rate);
    }
  } else {
    FOR_RANGE(int64_t, i, 0, n) {
      RmsPropUpdateFunctor<T, G, false>()(model_diff + i, model + i, n, scale, l1, l2,
                                          mean_square + i, nullptr, epsilon, weight_decay,
                                          decay_rate, *learning_rate);
    }
  }
}

template struct RmsPropUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct RmsPropUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename G>
struct LarsUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float momentum_beta,
                     float epsilon, float lars_coefficient, float weight_decay,
                     const float* learning_rate, const int64_t* train_step, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* momentum,
                     T* data_tmp, T* model_diff_tmp);
};

template<typename T, typename G>
void LarsUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float momentum_beta, float epsilon,
    float lars_coefficient, float weight_decay, const float* learning_rate,
    const int64_t* train_step, const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
    T* model, T* momentum, T* data_tmp, T* model_diff_tmp) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  T model_norm = data_tmp[0];
  T model_diff_norm = data_tmp[1];
  FOR_RANGE(int64_t, i, 0, n) {
    model_diff_tmp[i] =
        CastScaleRegularizeGradientFunctor<T, G>()(model_diff[i], model[i], scale, l1, l2);
  }
  KernelUtil<DeviceType::kCPU, T>::Dot(ctx, n, model, 1, model, 1, &model_norm);
  KernelUtil<DeviceType::kCPU, T>::Dot(ctx, n, model_diff_tmp, 1, model_diff_tmp, 1,
                                       &model_diff_norm);

  model_norm = std::sqrt(model_norm / n);
  model_diff_norm = std::sqrt(model_diff_norm / n);
  T local_learning_rate = 0;
  if (*train_step == 0) {
    local_learning_rate =
        *learning_rate * lars_coefficient * model_norm / (epsilon + model_diff_norm);
  } else {
    local_learning_rate = *learning_rate * lars_coefficient * model_norm
                          / (epsilon + model_diff_norm + weight_decay * model_norm);
  }

  FOR_RANGE(int64_t, i, 0, n) {
    LarsUpdateFunctor<T>()(model_diff_tmp + i, model + i, momentum_beta, momentum + i, weight_decay,
                           local_learning_rate);
  }
}

template struct LarsUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct LarsUpdateKernelUtil<DeviceType::kCPU, double, double>;

}  // namespace oneflow
