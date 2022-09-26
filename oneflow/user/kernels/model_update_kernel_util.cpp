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

namespace {

// For bias correction compute in CPU.
template<typename T>
T Fastpow(T a, int64_t b) {
  T ans = static_cast<T>(1);
  while (b) {
    if (b & 1) { ans *= a; }
    a *= a;
    b >>= 1;
  }
  return ans;
}

template<typename T>
void SumSquares2(int64_t n, const T* src0, T* dst0, const T* src1, T* dst1) {
  *dst0 += cblas_dot<T>(n, src0, 1, src0, 1);
  *dst1 += cblas_dot<T>(n, src1, 1, src1, 1);
}

}  // namespace

template<typename T, typename G, typename C>
struct SGDUpdateKernelUtil<DeviceType::kCPU, T, G, C> {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float weight_decay,
                     float learning_rate_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     C* model_copy);
};

template<typename T, typename G, typename C>
void SGDUpdateKernelUtil<DeviceType::kCPU, T, G, C>::Update(
    ep::Stream* stream, int64_t n, T scale, float l1, float l2, float weight_decay,
    float learning_rate_val, float lr_scale, const float* learning_rate, const T* scale_by_ptr,
    const int64_t* skip_if, const G* model_diff, T* model, C* model_copy) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  learning_rate_val *= lr_scale;
  for (int64_t i = 0; i != n; ++i) {
    if (model_copy != nullptr) {
      FusedSGDUpdateFunctor<T, G, C>()(model_diff + i, model + i, model_copy + i, scale, l1, l2,
                                       weight_decay, learning_rate_val);
    } else {
      SGDUpdateFunctor<T, G>()(model_diff + i, model + i, scale, l1, l2, weight_decay,
                               learning_rate_val);
    }
  }
}

template struct SGDUpdateKernelUtil<DeviceType::kCPU, float, float, float16>;
template struct SGDUpdateKernelUtil<DeviceType::kCPU, double, double, float16>;

template<typename T, typename K, typename IDX>
struct IndexedSlicesSGDUpdateKernelUtil<DeviceType::kCPU, T, K, IDX> {
  static void Update(ep::Stream* stream, float weight_decay, float lr_scale, int64_t num_indices,
                     int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                     const IDX* num_unique_instance, const float* learning_rate, const K* indices,
                     const T* values, T* model);
};

template<typename T, typename K, typename IDX>
void IndexedSlicesSGDUpdateKernelUtil<DeviceType::kCPU, T, K, IDX>::Update(
    ep::Stream* stream, float weight_decay, float lr_scale, int64_t num_indices,
    int64_t feature_size, int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
    const float* learning_rate, const K* indices, const T* values, T* model) {
  const int64_t n = *num_unique_instance * feature_size;
  T lr = *learning_rate;
  lr *= lr_scale;
  FOR_RANGE(int64_t, i, 0, n) {
    const IDX indices_idx = i / feature_size;
    const IDX inner_idx = i - indices_idx * feature_size;
    const IDX instance_id = indices[indices_idx];
    if (instance_id >= lower_bound && instance_id < upper_bound) {
      const IDX model_idx = (instance_id - lower_bound) * feature_size + inner_idx;
      SGDUpdateFunctor<T, T>()(values + i, model + model_idx, static_cast<T>(1), 0.0, 0.0,
                               weight_decay, lr);
    }
  }
}

#define INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_CPU(val_type_pair, key_type_pair,  \
                                                           idx_type_pair)                 \
  template struct IndexedSlicesSGDUpdateKernelUtil<                                       \
      DeviceType::kCPU, OF_PP_PAIR_FIRST(val_type_pair), OF_PP_PAIR_FIRST(key_type_pair), \
      OF_PP_PAIR_FIRST(idx_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_CPU,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INITIATE_INDEXED_SLICES_SGD_UPDATE_KERNEL_UTIL_CPU

template<typename T, typename G>
struct MomentumUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float beta,
                     float dampening, bool nesterov, bool maximize, float weight_decay,
                     float learning_rate_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     T* momentum);
};

template<typename T, typename G>
void MomentumUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    ep::Stream* stream, int64_t n, T scale, float l1, float l2, float beta, float dampening,
    bool nesterov, bool maximize, float weight_decay, float learning_rate_val, float lr_scale,
    const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
    T* model, T* momentum) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  learning_rate_val *= lr_scale;
  for (int64_t i = 0; i != n; ++i) {
    MomentumUpdateFunctor<T, G>()(model_diff + i, model + i, momentum + i, scale, l1, l2, beta,
                                  dampening, nesterov, maximize, weight_decay, learning_rate_val);
  }
}

template struct MomentumUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct MomentumUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename K, typename IDX>
struct IndexedSlicesMomentumMdUpdateKernelUtil<DeviceType::kCPU, T, K, IDX> {
  static void Update(ep::Stream* stream, T beta, float dampening, bool nesterov, bool maximize,
                     float weight_decay, float lr_scale, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const float* learning_rate, const K* indices, const T* values, T* model,
                     T* momentum);
};

template<typename T, typename K, typename IDX>
void IndexedSlicesMomentumMdUpdateKernelUtil<DeviceType::kCPU, T, K, IDX>::Update(
    ep::Stream* stream, T beta, float dampening, bool nesterov, bool maximize, float weight_decay,
    float lr_scale, int64_t num_instance, int64_t feature_size, int64_t lower_bound,
    int64_t upper_bound, const IDX* num_unique_instance, const float* learning_rate,
    const K* indices, const T* values, T* model, T* momentum) {
  const int64_t n = *num_unique_instance * feature_size;
  T lr = *learning_rate;
  lr *= lr_scale;
  for (int64_t i = 0; i != n; ++i) {
    const IDX indices_idx = i / feature_size;
    const IDX inner_idx = i - indices_idx * feature_size;
    const IDX instance_id = indices[indices_idx];
    if (instance_id >= lower_bound && instance_id < upper_bound) {
      const IDX model_idx = (instance_id - lower_bound) * feature_size + inner_idx;
      MomentumUpdateFunctor<T, T>()(values + i, model + model_idx, momentum + model_idx, 1.0, 0.0,
                                    0.0, beta, dampening, nesterov, maximize, weight_decay, lr);
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

template<typename T, typename G, typename C>
struct AdamUpdateKernelUtil<DeviceType::kCPU, T, G, C> {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, bool amsgrad,
                     bool do_bias_correction, float learning_rate_val, float lr_scale,
                     float bias_correction1_val, float bias_correction2_val,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     const float* bias_correction1, const float* bias_correction2,
                     const G* model_diff, T* model, C* model_copy, T* m, T* v, T* max_v);
};

template<typename T, typename G, typename C>
void AdamUpdateKernelUtil<DeviceType::kCPU, T, G, C>::Update(
    ep::Stream* stream, int64_t n, T scale, float l1, float l2, float beta1, float beta2,
    float epsilon, float weight_decay, bool amsgrad, bool do_bias_correction,
    float learning_rate_val, float lr_scale, float bias_correction1_val, float bias_correction2_val,
    const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
    const float* bias_correction1_ptr, const float* bias_correction2_ptr, const G* model_diff,
    T* model, C* model_copy, T* m, T* v, T* max_v) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  if (bias_correction1_ptr != nullptr) { bias_correction1_val = *bias_correction1_ptr; }
  if (bias_correction2_ptr != nullptr) { bias_correction2_val = *bias_correction2_ptr; }

  learning_rate_val *= lr_scale;
  FOR_RANGE(int64_t, i, 0, n) {
    if (model_copy != nullptr) {
      FusedAdamUpdateFunctor<T, G, C>()(model_diff + i, model + i, model_copy + i, m + i, v + i,
                                        max_v + i, scale, l1, l2, beta1, beta2, epsilon,
                                        weight_decay, amsgrad, bias_correction1_val,
                                        bias_correction2_val, learning_rate_val);
    } else {
      AdamUpdateFunctor<T, G>()(model_diff + i, model + i, m + i, v + i, max_v + i, scale, l1, l2,
                                beta1, beta2, epsilon, weight_decay, amsgrad, bias_correction1_val,
                                bias_correction2_val, learning_rate_val);
    }
  }
}

template struct AdamUpdateKernelUtil<DeviceType::kCPU, float, float, float16>;
template struct AdamUpdateKernelUtil<DeviceType::kCPU, double, double, float16>;

template<typename T, typename K, typename IDX>
struct IndexedSlicesAdamMdUpdateKernelUtil<DeviceType::kCPU, T, K, IDX> {
  static void Update(ep::Stream* stream, float beta1, float beta2, float epsilon,
                     float weight_decay, bool amsgrad, bool do_bias_correction, float lr,
                     float lr_scale, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const float* learning_rate, const float* bias_correction1_ptr,
                     const float* bias_correction2_ptr, const K* indices, const T* values, T* model,
                     T* m, T* v, T* max_v) {
    if (learning_rate != nullptr) { lr = *learning_rate; }
    lr *= lr_scale;
    float bias_correction1 = 1.0;
    float bias_correction2 = 1.0;
    if (bias_correction1_ptr != nullptr) { bias_correction1 = *bias_correction1_ptr; }
    if (bias_correction2_ptr != nullptr) { bias_correction2 = *bias_correction2_ptr; }

    const int64_t n = *num_unique_instance * feature_size;
    FOR_RANGE(int64_t, i, 0, n) {
      const IDX indices_idx = i / feature_size;
      const IDX inner_idx = i - indices_idx * feature_size;
      const IDX instance_id = indices[indices_idx];

      if (instance_id >= lower_bound && instance_id < upper_bound) {
        const IDX model_idx = (instance_id - lower_bound) * feature_size + inner_idx;
        AdamUpdateFunctor<T, T>()(values + i, model + model_idx, m + model_idx, v + model_idx,
                                  max_v + i, /*scale=*/1.0, /*l1=*/0.0, /*l2=*/0.0, beta1, beta2,
                                  epsilon, weight_decay, amsgrad, bias_correction1,
                                  bias_correction2, lr);
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
struct AdagradUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float lr_decay,
                     float epsilon, float weight_decay, float learning_rate_val, float lr_scale,
                     int64_t train_step, const float* learning_rate, const int64_t* train_step_ptr,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     T* sum);
};

template<typename T, typename G>
void AdagradUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    ep::Stream* stream, int64_t n, T scale, float l1, float l2, float lr_decay, float epsilon,
    float weight_decay, float learning_rate_val, float lr_scale, int64_t train_step,
    const float* learning_rate, const int64_t* train_step_ptr, const T* scale_by_ptr,
    const int64_t* skip_if, const G* model_diff, T* model, T* sum) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (train_step_ptr != nullptr) {
    train_step = *train_step_ptr + 1;
  }  // train_step_ptr start from zero.
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  learning_rate_val = learning_rate_val * lr_scale / (1 + (train_step - 1) * lr_decay);

  FOR_RANGE(int64_t, i, 0, n) {
    AdagradUpdateFunctor<T, G>()(model_diff + i, model + i, sum + i, scale, l1, l2, epsilon,
                                 weight_decay, learning_rate_val);
  }
}

template struct AdagradUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct AdagradUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename G>
struct LambUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(ep::Stream* stream, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, float learning_rate_val,
                     float lr_scale, bool do_bias_correction, float bias_correction1_val,
                     float bias_correction2_val, const float* learning_rate_ptr,
                     const float* bias_correction1_ptr, const float* bias_correction2_ptr,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
                     T* adam_diff, T* model, T* m, T* v, T* norm_buffer);
};

template<typename T, typename G>
void LambUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    ep::Stream* stream, int64_t n, float scale, float l1, float l2, float beta1, float beta2,
    float epsilon, float weight_decay, float learning_rate_val, float lr_scale,
    bool do_bias_correction, float bias_correction1_val, float bias_correction2_val,
    const float* learning_rate_ptr, const float* bias_correction1_ptr,
    const float* bias_correction2_ptr, const T* scale_by_ptr, const int64_t* skip_if,
    const G* model_diff, T* adam_diff, T* model, T* m, T* v, T* norm_buffer) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate_ptr != nullptr) { learning_rate_val = *learning_rate_ptr; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  if (bias_correction1_ptr != nullptr) { bias_correction1_val = *bias_correction1_ptr; }
  if (bias_correction2_ptr != nullptr) { bias_correction2_val = *bias_correction2_ptr; }

  FOR_RANGE(int64_t, i, 0, n) {
    LambGradFunctor<T, G>()(model_diff + i, adam_diff + i, model + i, m + i, v + i, scale, l1, l2,
                            beta1, beta2, epsilon, do_bias_correction, bias_correction1_val,
                            bias_correction2_val);
  }
  T* w_norm_2 = norm_buffer;
  T* g_norm_2 = norm_buffer + 1;
  Memset<DeviceType::kCPU>(stream, norm_buffer, 0, 2 * sizeof(T));
  SumSquares2(n, model, w_norm_2, adam_diff, g_norm_2);
  learning_rate_val *= lr_scale;
  const float lr = LambLRFunctor<T>()(learning_rate_val, w_norm_2, g_norm_2);
  FOR_RANGE(int64_t, i, 0, n) {
    LambUpdateFunctor<T>()(lr, weight_decay, adam_diff + i, model + i);
  }
}

template struct LambUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct LambUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<>
struct BiasCorrectionFactorKernelUtil<DeviceType::kCPU> {
  static void BiasCorrectionFactorCompute(ep::Stream* stream, float beta, const int64_t* train_step,
                                          float* out);
};

void BiasCorrectionFactorKernelUtil<DeviceType::kCPU>::BiasCorrectionFactorCompute(
    ep::Stream* stream, float beta, const int64_t* train_step, float* out) {
  const float bias_correction_factor = 1.0 - Fastpow<float>(beta, *train_step + 1);
  *out = bias_correction_factor;
}

template<typename T, typename G>
struct RmsPropUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, bool centered,
                     float epsilon, float weight_decay, float decay_rate, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* mean_square,
                     T* mean_gradient);
};

template<typename T, typename G>
void RmsPropUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    ep::Stream* stream, int64_t n, T scale, float l1, float l2, bool centered, float epsilon,
    float weight_decay, float decay_rate, float learning_rate_val, float lr_scale,
    const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
    T* model, T* mean_square, T* mean_gradient) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  learning_rate_val *= lr_scale;
  if (centered) {
    FOR_RANGE(int64_t, i, 0, n) {
      RmsPropUpdateFunctor<T, G, true>()(model_diff + i, model + i, n, scale, l1, l2,
                                         mean_square + i, mean_gradient + i, epsilon, weight_decay,
                                         decay_rate, learning_rate_val);
    }
  } else {
    FOR_RANGE(int64_t, i, 0, n) {
      RmsPropUpdateFunctor<T, G, false>()(model_diff + i, model + i, n, scale, l1, l2,
                                          mean_square + i, nullptr, epsilon, weight_decay,
                                          decay_rate, learning_rate_val);
    }
  }
}

template struct RmsPropUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct RmsPropUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename G>
struct LarsUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2,
                     float momentum_beta, float epsilon, float lars_coefficient, float weight_decay,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* momentum,
                     T* data_tmp, T* model_diff_tmp);
};

template<typename T, typename G>
void LarsUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    ep::Stream* stream, int64_t n, T scale, float l1, float l2, float momentum_beta, float epsilon,
    float lars_coefficient, float weight_decay, float lr_scale, const float* learning_rate,
    const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model, T* momentum,
    T* data_tmp, T* model_diff_tmp) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  T model_norm = data_tmp[0];
  T model_diff_norm = data_tmp[1];
  FOR_RANGE(int64_t, i, 0, n) {
    model_diff_tmp[i] =
        CastScaleRegularizeGradientFunctor<T, G>()(model_diff[i], model[i], scale, l1, l2);
  }
  Memset<DeviceType::kCPU>(stream, data_tmp, 0, 2 * sizeof(T));
  SumSquares2(n, model, &model_norm, model_diff_tmp, &model_diff_norm);
  model_norm = std::sqrt(model_norm);
  model_diff_norm = std::sqrt(model_diff_norm);
  T lars = static_cast<T>(1);
  if (model_norm > 0 && model_diff_norm > 0) {
    lars = lars_coefficient * model_norm / (epsilon + model_diff_norm + weight_decay * model_norm);
  }
  T lr = *learning_rate;
  lr *= lr_scale;
  T local_learning_rate = lr * lars;
  FOR_RANGE(int64_t, i, 0, n) {
    LarsUpdateFunctor<T>()(model_diff_tmp + i, model + i, momentum_beta, momentum + i, weight_decay,
                           local_learning_rate);
  }
}

template struct LarsUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct LarsUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename G>
struct FtrlUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float lr_power,
                     float lambda1, float lambda2, float beta, float weight_decay,
                     float learning_rate_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     T* accumulate, T* z);
};

template<typename T, typename G>
void FtrlUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    ep::Stream* stream, int64_t n, T scale, float l1, float l2, float lr_power, float lambda1,
    float lambda2, float beta, float weight_decay, float learning_rate_val, float lr_scale,
    const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
    T* model, T* accumulate, T* z) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  learning_rate_val *= lr_scale;
  for (int64_t i = 0; i != n; ++i) {
    FtrlUpdateFunctor<T, G>()(model_diff + i, model + i, accumulate + i, z + i, scale, l1, l2,
                              lr_power, lambda1, lambda2, beta, weight_decay, learning_rate_val);
  }
}

template struct FtrlUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct FtrlUpdateKernelUtil<DeviceType::kCPU, double, double>;

template<typename T, typename G>
struct AdadeltaUpdateKernelUtil<DeviceType::kCPU, T, G> {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float rho,
                     float epsilon, bool maximize, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* square_avgs,
                     T* acc_deltas);
};

template<typename T, typename G>
void AdadeltaUpdateKernelUtil<DeviceType::kCPU, T, G>::Update(
    ep::Stream* stream, int64_t n, T scale, float l1, float l2, float rho, float epsilon,
    bool maximize, float weight_decay, float learning_rate_val, float lr_scale,
    const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
    T* model, T* square_avgs, T* acc_deltas) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  learning_rate_val *= lr_scale;
  for (int64_t i = 0; i != n; ++i) {
    AdadeltaUpdateFunctor<T, G>()(model_diff + i, model + i, square_avgs + i, acc_deltas + i, scale,
                                  l1, l2, rho, epsilon, maximize, weight_decay, learning_rate_val);
  }
}

template struct AdadeltaUpdateKernelUtil<DeviceType::kCPU, float, float>;
template struct AdadeltaUpdateKernelUtil<DeviceType::kCPU, double, double>;

}  // namespace oneflow
