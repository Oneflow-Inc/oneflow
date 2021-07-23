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
#include "oneflow/user/kernels/ctc_loss_kernel_util.h"

namespace oneflow {

int get_target_prime(const int* targets_ptr, int64_t max_target_length, int64_t b, int64_t s,
                     int blank) {
  if (s % 2 == 0) {
    return blank;
  } else {
    int64_t idx = b * max_target_length + s / 2;
    return targets_ptr[idx];
  }
}

template<typename T, typename IDX>
struct CtcLossKernelUtil<DeviceType::kCPU, T, IDX> final {
  static void CtcLossForward(DeviceCtx* ctx, const T* log_probs_ptr, const int* targets_ptr,
                             const IDX* input_lengths_ptr, const IDX* target_lengths_ptr,
                             T* alpha_ptr, T* loss_ptr,
                             NdIndexOffsetHelper<int64_t, 3>& input_helper,
                             NdIndexOffsetHelper<int64_t, 3>& alpha_helper,
                             const int64_t batch_size, const int64_t max_input_length,
                             const int64_t max_target_length, const int blank);

  static void CtcLossBackward(DeviceCtx* ctx, const T* grad_out_ptr, const T* loss_ptr,
                              const T* alpha_ptr, const T* log_probs_ptr, const int* targets_ptr,
                              const IDX* input_lengths_ptr, const IDX* target_lengths_ptr,
                              T* beta_ptr, T* grad_ptr,
                              NdIndexOffsetHelper<int64_t, 3>& input_helper,
                              NdIndexOffsetHelper<int64_t, 3>& beta_helper,
                              const int64_t batch_size, const int64_t max_input_length,
                              const int64_t max_target_length, const int64_t num_labels,
                              const int blank, const bool zero_infinity);
};

template<typename T, typename IDX>
void CtcLossKernelUtil<DeviceType::kCPU, T, IDX>::CtcLossForward(
    DeviceCtx* ctx, const T* log_probs_ptr, const int* targets_ptr, const IDX* input_lengths_ptr,
    const IDX* target_lengths_ptr, T* alpha_ptr, T* loss_ptr,
    NdIndexOffsetHelper<int64_t, 3>& input_helper, NdIndexOffsetHelper<int64_t, 3>& alpha_helper,
    const int64_t batch_size, const int64_t max_input_length, const int64_t max_target_length,
    const int blank) {
  constexpr T neginf = -std::numeric_limits<T>::infinity();
  FOR_RANGE(int64_t, b, 0, batch_size) {
    CHECK_GE(max_input_length, input_lengths_ptr[b]);
    CHECK_GE(max_target_length, target_lengths_ptr[b]);
  }
  FOR_RANGE(int32_t, b, 0, batch_size) {
    IDX input_length = input_lengths_ptr[b];
    IDX target_length = target_lengths_ptr[b];

    int64_t alpha_idx = alpha_helper.NdIndexToOffset(b, 0, 0);
    for (IDX s = 0; s < 2 * target_length + 1; s++) { alpha_ptr[alpha_idx + s] = neginf; }
    alpha_ptr[alpha_idx] = log_probs_ptr[input_helper.NdIndexToOffset(0, b, blank)];
    if (target_length > 0) {
      int target = get_target_prime(targets_ptr, max_target_length, b, 1, blank);
      alpha_ptr[alpha_idx + 1] = log_probs_ptr[input_helper.NdIndexToOffset(0, b, target)];
    }

    for (IDX t = 1; t < input_length; t++) {
      for (IDX s = 0; s < 2 * target_length + 1; s++) {
        int current_target_prime = get_target_prime(targets_ptr, max_target_length, b, s, blank);
        T la1 = alpha_ptr[alpha_helper.NdIndexToOffset(b, t - 1, s)];
        T la2, la3, lamax = la1;
        if (s > 0) {
          la2 = alpha_ptr[alpha_helper.NdIndexToOffset(b, t - 1, s - 1)];
          if (la2 > lamax) lamax = la2;
        } else {
          la2 = neginf;
        }
        if ((s > 1)
            && (get_target_prime(targets_ptr, max_target_length, b, s - 2, blank)
                != current_target_prime)) {
          la3 = alpha_ptr[alpha_helper.NdIndexToOffset(b, t - 1, s - 2)];
          if (la3 > lamax) lamax = la3;
        } else {
          la3 = neginf;
        }
        if (lamax == neginf) lamax = 0;

        int64_t idx_t_s = alpha_helper.NdIndexToOffset(b, t, s);
        alpha_ptr[idx_t_s] =
            std::log(std::exp(la1 - lamax) + std::exp(la2 - lamax) + std::exp(la3 - lamax)) + lamax
            + log_probs_ptr[input_helper.NdIndexToOffset(t, b, current_target_prime)];
      }
    }

    if (target_length == 0) {
      int64_t idx = alpha_helper.NdIndexToOffset(b, input_length - 1, 0);
      loss_ptr[b] = -alpha_ptr[idx];
    } else {
      int64_t idx1 = alpha_helper.NdIndexToOffset(b, input_length - 1, target_length * 2);
      int64_t idx2 = alpha_helper.NdIndexToOffset(b, input_length - 1, target_length * 2 - 1);
      T l1 = alpha_ptr[idx1];
      T l2 = alpha_ptr[idx2];
      T m = std::max(l1, l2);
      m = ((m == neginf) ? 0 : m);
      T log_likelihood = std::log(std::exp(l1 - m) + std::exp(l2 - m)) + m;
      loss_ptr[b] = -log_likelihood;
    }
  }
}

template<typename T, typename IDX>
void CtcLossKernelUtil<DeviceType::kCPU, T, IDX>::CtcLossBackward(
    DeviceCtx* ctx, const T* grad_out_ptr, const T* loss_ptr, const T* alpha_ptr,
    const T* log_probs_ptr, const int* targets_ptr, const IDX* input_lengths_ptr,
    const IDX* target_lengths_ptr, T* beta_ptr, T* grad_ptr,
    NdIndexOffsetHelper<int64_t, 3>& input_helper, NdIndexOffsetHelper<int64_t, 3>& beta_helper,
    const int64_t batch_size, const int64_t max_input_length, const int64_t max_target_length,
    const int64_t num_labels, const int blank, const bool zero_infinity) {
  constexpr T neginf = -std::numeric_limits<T>::infinity();
  int64_t elem_cnt = max_input_length * batch_size * num_labels;
  FOR_RANGE(int64_t, i, 0, elem_cnt) { grad_ptr[i] = neginf; }

  FOR_RANGE(int64_t, b, 0, batch_size) {
    IDX input_length = input_lengths_ptr[b];
    IDX target_length = target_lengths_ptr[b];
    T nll = loss_ptr[b];
    if (zero_infinity && nll == std::numeric_limits<T>::infinity()) {
      for (IDX t = 0; t < max_input_length; t++) {
        for (IDX c = 0; c < num_labels; c++) {
          grad_ptr[input_helper.NdIndexToOffset(t, b, c)] = 0;
        }
      }
      continue;
    }

    if (input_length > 0) {
      int64_t beta_idx = beta_helper.NdIndexToOffset(b, input_length - 1, 0);
      for (IDX s = 0; s < 2 * target_length + 1; s++) { beta_ptr[beta_idx + s] = neginf; }
      beta_ptr[beta_idx + 2 * target_length] =
          log_probs_ptr[input_helper.NdIndexToOffset(input_length - 1, b, blank)];
      grad_ptr[input_helper.NdIndexToOffset(input_length - 1, b, blank)] =
          alpha_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length)]
          + beta_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length)];

      if (target_length > 0) {
        int target =
            get_target_prime(targets_ptr, max_target_length, b, 2 * target_length - 1, blank);
        beta_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length - 1)] =
            log_probs_ptr[input_helper.NdIndexToOffset(input_length - 1, b, target)];
        grad_ptr[input_helper.NdIndexToOffset(input_length - 1, b, target)] =
            alpha_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length - 1)]
            + beta_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length - 1)];
      }
    }

    for (IDX t = input_length - 2; t >= 0; t--) {
      for (IDX s = 2 * target_length; s >= 0; s--) {
        int current_target_prime = get_target_prime(targets_ptr, max_target_length, b, s, blank);
        T lb1 = beta_ptr[beta_helper.NdIndexToOffset(b, t + 1, s)];
        T lb2, lb3, lbmax = lb1;

        if (s < 2 * target_length) {
          lb2 = beta_ptr[beta_helper.NdIndexToOffset(b, t + 1, s + 1)];
          if (lb2 > lbmax) lbmax = lb2;
        } else {
          lb2 = neginf;
        }

        if ((s < 2 * target_length - 1)
            && (get_target_prime(targets_ptr, max_target_length, b, s + 2, blank)
                != current_target_prime)) {
          lb3 = beta_ptr[beta_helper.NdIndexToOffset(b, t + 1, s + 2)];
          if (lb3 > lbmax) lbmax = lb3;
        } else {
          lb3 = neginf;
        }
        if (lbmax == neginf) lbmax = 0;

        int64_t idx_t_s = beta_helper.NdIndexToOffset(b, t, s);
        beta_ptr[idx_t_s] =
            std::log(std::exp(lb1 - lbmax) + std::exp(lb2 - lbmax) + std::exp(lb3 - lbmax)) + lbmax
            + log_probs_ptr[input_helper.NdIndexToOffset(t, b, current_target_prime)];

        T log_alpha_beta = alpha_ptr[idx_t_s] + beta_ptr[idx_t_s];
        T& lcab = grad_ptr[input_helper.NdIndexToOffset(t, b, current_target_prime)];
        if (lcab == neginf) {
          lcab = log_alpha_beta;
        } else {
          T m = std::max(lcab, log_alpha_beta);
          lcab = std::log(std::exp(lcab - m) + std::exp(log_alpha_beta - m)) + m;
        }
      }
    }

    for (int32_t t = 0; t < input_length; t++) {
      for (int64_t c = 0; c < num_labels; c++) {
        T& res = grad_ptr[input_helper.NdIndexToOffset(t, b, c)];
        T lp = log_probs_ptr[input_helper.NdIndexToOffset(t, b, c)];
        res = (std::exp(lp) - std::exp(res + nll - lp)) * grad_out_ptr[b];
      }
    }

    // zero the remainder
    if (input_length < max_input_length) {
      for (int64_t t = input_length; t < max_input_length; t++) {
        for (int64_t c = 0; c < num_labels; c++) {
          int64_t grad_idx = input_helper.NdIndexToOffset(t, b, c);
          grad_ptr[grad_idx] = 0;
        }
      }
    }
  }
}

#define INSTANTIATE_CTC_LOSS_KERNEL_UTIL_CPU(device_type_v, log_probs_dtype_pair,          \
                                             input_lengths_dtype_pair)                     \
  template struct CtcLossKernelUtil<device_type_v, OF_PP_PAIR_FIRST(log_probs_dtype_pair), \
                                    OF_PP_PAIR_FIRST(input_lengths_dtype_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CTC_LOSS_KERNEL_UTIL_CPU, (DeviceType::kCPU),
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef INSTANTIATE_CTC_LOSS_KERNEL_UTIL_CPU

}  // namespace oneflow
