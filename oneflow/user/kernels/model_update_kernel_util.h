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
#ifndef ONEFLOW_USER_KERNELS_MODEL_UPDATE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_MODEL_UPDATE_KERNEL_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/math_unary_elementwise_func.h"

namespace oneflow {

template<typename T, typename G>
struct CastScaleRegularizeGradientFunctor {
  OF_DEVICE_FUNC
  T operator()(G model_diff, T model, T scale, float l1, float l2) const {
    return static_cast<T>(model_diff) * scale + l1 * ((model >= 0) - (model <= 0)) + l2 * model;
  }
};

template<typename T, typename G>
struct SGDUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, T scale, float l1, float l2, float weight_decay,
                  float learning_rate) const {
    const T model_val = *model;
    const T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);
    const T next_model = model_val - learning_rate * (model_diff_t + weight_decay * model_val);
    *model = next_model;
  }
};

template<DeviceType device_type, typename T, typename G>
struct SGDUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float weight_decay,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     const G* model_diff, T* model);
};

template<DeviceType device_type, typename T, typename K>
struct IndexedSlicesSGDUpdateKernelUtil final {
  static void Update(DeviceCtx* ctx, int64_t num_indices, int64_t num_features,
                     int64_t feature_size, int64_t feature_id_offset, const float* learning_rate,
                     const K* indices, const T* values, T* model);
};

template<typename T, typename G>
struct MomentumUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, T* momentum, T scale, float l1, float l2,
                  float beta, float weight_decay, float learning_rate) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);
    const T next_momentum = beta * *momentum - learning_rate * model_diff_t;
    *momentum = next_momentum;
    const T next_model = model_val + next_momentum - learning_rate * weight_decay * model_val;
    *model = next_model;
  }
};

template<typename T, typename G>
struct AdamUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, T* m, T* v, T scale, float l1, float l2,
                  float beta1, float beta2, float epsilon, float weight_decay,
                  float learning_rate) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);
    const T next_m = beta1 * *m + (1 - beta1) * model_diff_t;
    *m = next_m;
    const T next_v = beta2 * *v + (1 - beta2) * model_diff_t * model_diff_t;
    *v = next_v;
    *model =
        model_val - learning_rate * (next_m / (sqrt(next_v) + epsilon) + weight_decay * model_val);
  }
};

template<typename T, typename G>
struct LambGradFunctor {
  OF_DEVICE_FUNC
  void operator()(const T* beta1_t, const T* beta2_t, const G* model_diff, T* adam_diff, T* model,
                  T* m, T* v, float scale, float l1, float l2, float beta1, float beta2,
                  float epsilon) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);
    const T next_m = beta1 * *m + (1 - beta1) * model_diff_t;
    const T next_v = beta2 * *v + (1 - beta2) * model_diff_t * model_diff_t;
    *adam_diff = (next_m / (1 - *beta1_t)) / std::sqrt(next_v / (1 - *beta2_t) + epsilon);
    *m = next_m;
    *v = next_v;
  }
};

template<typename T>
struct LambLRFunctor {
  OF_DEVICE_FUNC
  float operator()(const float learning_rate, const T* w_norm, const T* g_norm) const {
    float lr = learning_rate;
    const T w_norm_val = *w_norm;
    const T g_norm_val = *g_norm;
    T trust_ratio = 1;
    if (w_norm_val > 0 && g_norm_val > 0) { trust_ratio = w_norm_val / g_norm_val; }
    lr *= trust_ratio;
    return lr;
  }
};

template<typename T>
struct LambUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const float learning_rate, const float weight_decay, const T* adam_diff,
                  T* model) const {
    const T model_val = *model;
    *model = model_val - learning_rate * (*adam_diff + weight_decay * model_val);
  }
};

template<DeviceType device_type, typename T, typename G>
struct MomentumUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta,
                     float weight_decay, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* momentum);
};

template<DeviceType device_type, typename T, typename K, typename IDX>
struct IndexedSlicesMomentumMdUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, T beta, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const float* learning_rate, const K* indices, const T* values, T* model,
                     T* momentum);
};

template<DeviceType device_type, typename T, typename G>
struct AdamUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     T* m, T* v);
};

template<DeviceType device_type, typename T, typename K, typename IDX>
struct IndexedSlicesAdamMdUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, float beta1, float beta2, float epsilon, int64_t num_instance,
                     int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                     const IDX* num_unique_instance, const float* learning_rate, const K* indices,
                     const T* values, T* model, T* m, T* v);
};

template<DeviceType device_type, typename T, typename G>
struct LambUpdateKernelUtil {
 public:
  static void Update(DeviceCtx* ctx, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
                     T* adam_diff, T* model, T* m, T* v, T* norm_buffer, T* beta1_t, T* beta2_t);
};

template<DeviceType device_type>
struct AdamBiasCorrectionLearningRateKernelUtil {
 public:
  static void AdamBiasCorrectionLearningRate(DeviceCtx* ctx, float beta1, float beta2,
                                             const float* learning_rate, const int64_t* train_step,
                                             float* out);
};

template<typename T, typename G, bool centered>
struct RmsPropUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, int64_t n, T scale, float l1, float l2,
                  T* mean_square, T* mean_gradient, float epsilon, float weight_decay,
                  float decay_rate, const float learning_rate) const {
    const T model_val = *model;
    T model_diff_t = CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, *model, scale, l1, l2);
    T mean_square_val = *mean_square;
    mean_square_val = (1 - decay_rate) * model_diff_t * model_diff_t + decay_rate * mean_square_val;
    *mean_square = mean_square_val;
    T denom_t;
    if (centered) {
      T mean_gradient_val = *mean_gradient;
      mean_gradient_val = (1 - decay_rate) * model_diff_t + decay_rate * mean_gradient_val;
      *mean_gradient = mean_gradient_val;
      denom_t = mean_square_val - mean_gradient_val * mean_gradient_val;
    } else {
      denom_t = *mean_square;
    }
    *model = model_val - learning_rate * model_diff_t * RsqrtFunctor<T>::Forward(denom_t + epsilon);
  }
};

template<DeviceType device_type, typename T, typename G>
struct RmsPropUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, bool centered,
                     float epsilon, float weight_decay, float decay_rate,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     const G* model_diff, T* model, T* mean_square, T* mean_gradient);
};

template<typename T>
struct LarsUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(T* model_diff_tmp, T* model, float momentum_beta, T* momentum, float weight_decay,
                  const T local_learning_rate) const {
    const T model_val = *model;
    T reg_diff = *model_diff_tmp + *model * weight_decay;
    T next_momentum = *momentum * momentum_beta - local_learning_rate * reg_diff;
    *momentum = next_momentum;
    *model = model_val + next_momentum;
  }
};

template<DeviceType device_type, typename T, typename G>
struct LarsUpdateKernelUtil {
  static void Update(DeviceCtx* ctx, int64_t n, T scale, float l1, float l2, float momentum_beta,
                     float epsilon, float lars_coefficient, float weight_decay,
                     const float* learning_rate, const int64_t* train_step, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* momentum,
                     T* data_tmp, T* model_diff_tmp);
};

#endif

}  // namespace oneflow
