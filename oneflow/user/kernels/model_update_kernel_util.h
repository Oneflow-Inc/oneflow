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

template<typename T, typename G, typename C>
struct FusedSGDUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, C* model_copy, T scale, float l1, float l2,
                  float weight_decay, float learning_rate) const {
    const T model_val = *model;
    const T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);
    const T next_model = model_val - learning_rate * (model_diff_t + weight_decay * model_val);
    *model = next_model;
    *model_copy = static_cast<C>(next_model);
  }
};

template<DeviceType device_type, typename T, typename G, typename C>
struct SGDUpdateKernelUtil {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float weight_decay,
                     float learning_rate_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     C* model_copy);
};

template<DeviceType device_type, typename T, typename K, typename IDX>
struct IndexedSlicesSGDUpdateKernelUtil final {
  static void Update(ep::Stream* stream, float weight_decay, float lr_scale, int64_t num_indices,
                     int64_t feature_size, int64_t lower_bound, int64_t upper_bound,
                     const IDX* num_unique_instance, const float* learning_rate, const K* indices,
                     const T* values, T* model);
};

template<typename T, typename G>
struct MomentumUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, T* momentum, T scale, float l1, float l2,
                  float beta, float dampening, bool nesterov, bool maximize, float weight_decay,
                  float learning_rate) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);

    T next_momentum = beta * *momentum + (1.0f - dampening) * model_diff_t;
    *momentum = next_momentum;

    if (!nesterov) {
      model_diff_t = next_momentum;
    } else {
      model_diff_t += beta * next_momentum;
    }

    T alpha = -learning_rate;
    if (maximize) { alpha = learning_rate; }
    const T next_model =
        model_val + alpha * model_diff_t - learning_rate * weight_decay * model_val;
    *model = next_model;
  }
};

template<typename T, typename G>
struct AdamUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, T* m, T* v, T* max_v, T scale, float l1, float l2,
                  float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad,
                  float bias_correction1, float bias_correction2, float learning_rate) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);

    const T next_m = beta1 * *m + (1 - beta1) * model_diff_t;
    *m = next_m;

    const T next_v = beta2 * *v + (1 - beta2) * model_diff_t * model_diff_t;
    *v = next_v;

    T denom = 0;
    if (amsgrad) {
      const T next_max_v =
          *max_v > next_v ? *max_v : next_v;  // use std::max has bug in GPU kernel.
      *max_v = next_max_v;
      denom = (sqrt(next_max_v) / sqrt(bias_correction2)) + epsilon;
    } else {
      denom = (sqrt(next_v) / sqrt(bias_correction2)) + epsilon;
    }
    const T step_size = learning_rate / bias_correction1;
    *model = model_val - step_size * (next_m / denom) - learning_rate * weight_decay * model_val;
  }
};

template<typename T, typename G, typename C>
struct FusedAdamUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, C* model_copy, T* m, T* v, T* max_v, T scale,
                  float l1, float l2, float beta1, float beta2, float epsilon, float weight_decay,
                  bool amsgrad, float bias_correction1, float bias_correction2,
                  float learning_rate) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);

    const T next_m = beta1 * *m + (1 - beta1) * model_diff_t;
    *m = next_m;

    const T next_v = beta2 * *v + (1 - beta2) * model_diff_t * model_diff_t;
    *v = next_v;

    T denom = 0;
    if (amsgrad) {
      const T next_max_v =
          *max_v > next_v ? *max_v : next_v;  // use std::max has bug in GPU kernel.
      *max_v = next_max_v;
      denom = (sqrt(next_max_v) / sqrt(bias_correction2)) + epsilon;
    } else {
      denom = (sqrt(next_v) / sqrt(bias_correction2)) + epsilon;
    }
    const T step_size = learning_rate / bias_correction1;
    const T next_model =
        model_val - step_size * (next_m / denom) - learning_rate * weight_decay * model_val;
    *model = next_model;
    *model_copy = static_cast<C>(next_model);
  }
};

template<typename T, typename G>
struct AdagradUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* model, T* sum, T scale, float l1, float l2, float epsilon,
                  float weight_decay, float learning_rate) {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);
    const T next_sum = *sum + model_diff_t * model_diff_t;
    *sum = next_sum;
    *model = model_val - learning_rate / (sqrt(next_sum) + epsilon) * model_diff_t
             - learning_rate * weight_decay * model_val;
  }
};

template<typename T, typename G>
struct LambGradFunctor {
  OF_DEVICE_FUNC
  void operator()(const G* model_diff, T* adam_diff, T* model, T* m, T* v, float scale, float l1,
                  float l2, float beta1, float beta2, float epsilon, bool do_bias_correction,
                  float bias_correction1, float bias_correction2) const {
    const T model_val = *model;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);
    const T next_m = beta1 * *m + (1 - beta1) * model_diff_t;
    const T next_v = beta2 * *v + (1 - beta2) * model_diff_t * model_diff_t;
    *m = next_m;
    *v = next_v;
    T numerator = 0;
    T denominator = 0;
    if (do_bias_correction) {
      numerator = next_m / bias_correction1;
      denominator = (sqrt(next_v) / sqrt(bias_correction2)) + epsilon;
    } else {
      numerator = next_m;
      denominator = sqrt(next_v) + epsilon;
    }
    *adam_diff = numerator / denominator;
  }
};

template<typename T>
struct LambLRFunctor {
  OF_DEVICE_FUNC
  float operator()(const float learning_rate_val, const T* w_norm_2, const T* g_norm_2) const {
    float lr = learning_rate_val;
    const T w_norm_val = sqrt(*w_norm_2);
    const T g_norm_val = sqrt(*g_norm_2);
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

template<typename T, typename G>
struct FtrlUpdateFunctor {
  OF_DEVICE_FUNC void operator()(const G* model_diff, T* model, T* accumulate, T* z, T scale,
                                 float l1, float l2, float lr_power, float lambda1, float lambda2,
                                 float beta, float weight_decay, float learning_rate) {
    const T model_val = *model;
    const T z_val = *z;
    const float lr_reciprocal = static_cast<float>(1.0) / learning_rate;
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(*model_diff, model_val, scale, l1, l2);
    const T accumulate_val = *accumulate;
    const T next_accumulate_val = accumulate_val + model_diff_t * model_diff_t;
    const T acc_powered = pow(accumulate_val, lr_power);
    const T next_acc_powered = pow(next_accumulate_val, lr_power);
    const T sigma = (next_acc_powered - acc_powered) * lr_reciprocal;
    const T new_z_val = z_val + model_diff_t - sigma * model_val;
    T new_model = static_cast<T>(0.0);
    if (abs(new_z_val) >= lambda1) {
      new_model = (copysign(lambda1, new_z_val) - new_z_val)
                      / ((beta + next_acc_powered) * lr_reciprocal + lambda2)
                  - learning_rate * weight_decay * model_val;
    }
    *model = new_model;
    *accumulate = next_accumulate_val;
    *z = new_z_val;
  }
};

template<typename T, typename G>
struct AdadeltaUpdateFunctor {
  OF_DEVICE_FUNC void operator()(const G* model_diff, T* model, T* square_avgs, T* acc_deltas,
                                 T scale, float l1, float l2, float rho, float epsilon,
                                 bool maximize, float weight_decay, float learning_rate) {
    const T model_val = *model;
    T model_diff_val = *model_diff;
    if (maximize) { model_diff_val = -model_diff_val; }
    T model_diff_t =
        CastScaleRegularizeGradientFunctor<T, G>()(model_diff_val, model_val, scale, l1, l2);
    T square_avgs_val = *square_avgs;
    T new_square_avgs_val = square_avgs_val * rho + model_diff_t * model_diff_t * (1.0f - rho);
    T square_avgs_std = sqrt(new_square_avgs_val + epsilon);
    T acc_delta_val = *acc_deltas;
    T delta = sqrt(acc_delta_val + epsilon) / square_avgs_std * model_diff_t;
    T new_acc_deltas = acc_delta_val * rho + delta * delta * (1.0f - rho);
    T new_model = model_val - learning_rate * delta;
    *model = new_model;
    *square_avgs = new_square_avgs_val;
    *acc_deltas = new_acc_deltas;
  }
};

template<DeviceType device_type>
struct BiasCorrectionFactorKernelUtil {
 public:
  static void BiasCorrectionFactorCompute(ep::Stream* stream, float beta, const int64_t* train_step,
                                          float* out);
};

template<DeviceType device_type, typename T, typename G>
struct MomentumUpdateKernelUtil {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float beta,
                     float dampening, bool nesterov, bool maximize, float weight_decay,
                     float learning_rate_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     T* momentum);
};

template<DeviceType device_type, typename T, typename K, typename IDX>
struct IndexedSlicesMomentumMdUpdateKernelUtil {
  static void Update(ep::Stream* stream, T beta, float dampening, bool nesterov, bool maximize,
                     float weight_decay, float lr_scale, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const float* learning_rate, const K* indices, const T* values, T* model,
                     T* momentum);
};

template<DeviceType device_type, typename T, typename G, typename C>
struct AdamUpdateKernelUtil {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, bool amsgrad,
                     bool do_bias_correction, float learning_rate_val, float lr_scale,
                     float bias_correction1_val, float bias_correction2_val,
                     const float* learning_rate, const T* scale_by_ptr, const int64_t* skip_if,
                     const float* bias_correction1, const float* bias_correction2,
                     const G* model_diff, T* model, C* model_copy, T* m, T* v, T* max_v);
};

template<DeviceType device_type, typename T, typename G>
struct AdagradUpdateKernelUtil {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float lr_decay,
                     float epsilon, float weight_decay, float learning_rate_val, float lr_scale,
                     int64_t train_step, const float* learning_rate, const int64_t* train_step_ptr,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     T* sum);
};

template<DeviceType device_type, typename T, typename K, typename IDX>
struct IndexedSlicesAdamMdUpdateKernelUtil {
  static void Update(ep::Stream* stream, float beta1, float beta2, float epsilon,
                     float weight_decay, bool amsgrad, bool do_bias_correction, float lr,
                     float lr_scale, int64_t num_instance, int64_t feature_size,
                     int64_t lower_bound, int64_t upper_bound, const IDX* num_unique_instance,
                     const float* learning_rate, const float* bias_correction1_ptr,
                     const float* bias_correction2_ptr, const K* indices, const T* values, T* model,
                     T* m, T* v, T* max_v);
};

template<DeviceType device_type, typename T, typename G>
struct LambUpdateKernelUtil {
 public:
  static void Update(ep::Stream* stream, int64_t n, float scale, float l1, float l2, float beta1,
                     float beta2, float epsilon, float weight_decay, float learning_rate_val,
                     float lr_scale, bool do_bias_correction, float bias_correction1_val,
                     float bias_correction2_val, const float* learning_rate_ptr,
                     const float* bias_correction1_ptr, const float* bias_correction2_ptr,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff,
                     T* adam_diff, T* model, T* m, T* v, T* norm_buffer);
};

template<DeviceType device_type, typename T, typename G>
struct FtrlUpdateKernelUtil {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float lr_power,
                     float lambda1, float lambda2, float beta, float weight_decay,
                     float learning_rate_val, float lr_scale, const float* learning_rate,
                     const T* scale_by_ptr, const int64_t* skip_if, const G* model_diff, T* model,
                     T* accumulate, T* z);
};

template<DeviceType device_type, typename T, typename G>
struct AdadeltaUpdateKernelUtil {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, float rho,
                     float epsilon, bool maximize, float weight_decay, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* square_avgs,
                     T* acc_deltas);
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
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2, bool centered,
                     float epsilon, float weight_decay, float decay_rate, float learning_rate_val,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* mean_square,
                     T* mean_gradient);
};

template<typename T>
struct LarsUpdateFunctor {
  OF_DEVICE_FUNC
  void operator()(T* model_diff_tmp, T* model, float momentum_beta, T* momentum, float weight_decay,
                  const T local_learning_rate) const {
    const T model_val = *model;
    T next_momentum = *momentum * momentum_beta - local_learning_rate * *model_diff_tmp;
    *momentum = next_momentum;
    const T next_model = model_val + next_momentum - local_learning_rate * weight_decay * model_val;
    *model = next_model;
  }
};

template<DeviceType device_type, typename T, typename G>
struct LarsUpdateKernelUtil {
  static void Update(ep::Stream* stream, int64_t n, T scale, float l1, float l2,
                     float momentum_beta, float epsilon, float lars_coefficient, float weight_decay,
                     float lr_scale, const float* learning_rate, const T* scale_by_ptr,
                     const int64_t* skip_if, const G* model_diff, T* model, T* momentum,
                     T* data_tmp, T* model_diff_tmp);
};

#endif

}  // namespace oneflow
