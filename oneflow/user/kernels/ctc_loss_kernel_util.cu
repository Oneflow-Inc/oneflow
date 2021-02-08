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

namespace {

__device__ __inline__ static int get_target_prime(const int* targets_ptr, int64_t max_target_length,
                                                  int64_t b, int64_t s, int blank) {
  if (s % 2 == 0) {
    return blank;
  } else {
    int64_t idx = b * max_target_length + s / 2;
    return targets_ptr[idx];
  }
}

template<typename T, typename IDX>
__global__ void CtcLossGpu(const T* log_probs_ptr, const int* targets_ptr,
                           const IDX* input_lengths_ptr, const IDX* target_lengths_ptr,
                           T* alpha_ptr, T* loss_ptr, NdIndexOffsetHelper<int64_t, 3> input_helper,
                           NdIndexOffsetHelper<int64_t, 3> alpha_helper, const int64_t batch_size,
                           const int64_t max_input_length, const int64_t max_target_length,
                           const int blank) {
  constexpr T neginf = -INFINITY;
  const int32_t bid = blockIdx.x;
  const int32_t tid = threadIdx.x;
  for (int64_t b = bid; b < batch_size; b += gridDim.x) {
    if (tid == 0) {
      if (input_lengths_ptr[b] > max_input_length) __trap();
      if (target_lengths_ptr[b] > max_target_length) __trap();
    }
  }
  for (int64_t b = bid; b < batch_size; b += gridDim.x) {
    IDX input_length = input_lengths_ptr[b];
    IDX target_length = target_lengths_ptr[b];

    for (IDX s = tid; s < 2 * target_length + 1; s += blockDim.x) {
      alpha_ptr[alpha_helper.NdIndexToOffset(b, 0, s)] = neginf;
    }
    if (tid == 0) {
      alpha_ptr[alpha_helper.NdIndexToOffset(b, 0, 0)] =
          log_probs_ptr[input_helper.NdIndexToOffset(0, b, blank)];
      if (target_length > 0) {
        int target = get_target_prime(targets_ptr, max_target_length, b, 1, blank);
        alpha_ptr[alpha_helper.NdIndexToOffset(b, 0, 1)] =
            log_probs_ptr[input_helper.NdIndexToOffset(0, b, target)];
      }
    }
    __syncthreads();
    for (IDX t = 1; t < input_length; t++) {
      for (IDX s = tid; s < 2 * target_length + 1; s += blockDim.x) {
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
            log(exp(la1 - lamax) + exp(la2 - lamax) + exp(la3 - lamax)) + lamax
            + log_probs_ptr[input_helper.NdIndexToOffset(t, b, current_target_prime)];
      }
      __syncthreads();
    }
    if (tid == 0) {
      if (target_length == 0) {
        int64_t idx = alpha_helper.NdIndexToOffset(b, input_length - 1, 0);
        loss_ptr[b] = -alpha_ptr[idx];
      } else {
        int64_t idx1 = alpha_helper.NdIndexToOffset(b, input_length - 1, target_length * 2);
        int64_t idx2 = alpha_helper.NdIndexToOffset(b, input_length - 1, target_length * 2 - 1);
        T l1 = alpha_ptr[idx1];
        T l2 = alpha_ptr[idx2];
        T m = max(l1, l2);
        m = ((m == neginf) ? 0 : m);
        T log_likelihood = log(exp(l1 - m) + exp(l2 - m)) + m;
        loss_ptr[b] = -log_likelihood;
      }
    }
  }
}

template<typename T, typename IDX>
__global__ void CtcLossGradGpu(const T* grad_out_ptr, const T* loss_ptr, const T* alpha_ptr,
                               const T* log_probs_ptr, const int* targets_ptr,
                               const IDX* input_lengths_ptr, const IDX* target_lengths_ptr,
                               T* beta_ptr, T* grad_ptr,
                               NdIndexOffsetHelper<int64_t, 3> input_helper,
                               NdIndexOffsetHelper<int64_t, 3> beta_helper,
                               const int64_t batch_size, const int64_t max_input_length,
                               const int64_t max_target_length, const int64_t num_labels,
                               const int blank, const bool zero_infinity) {
  constexpr T neginf = -INFINITY;
  const int32_t bid = blockIdx.x;
  const int32_t tid = threadIdx.x;

  for (int64_t b = bid; b < batch_size; b += gridDim.x) {
    IDX input_length = input_lengths_ptr[b];
    IDX target_length = target_lengths_ptr[b];
    T nll = loss_ptr[b];
    if (zero_infinity && nll == INFINITY) {
      for (IDX t = tid; t < max_input_length; t += blockDim.x) {
        for (IDX c = 0; c < num_labels; c++) {
          grad_ptr[input_helper.NdIndexToOffset(t, b, c)] = 0;
        }
      }
      __syncthreads();
      continue;
    }

    if (input_length > 0) {
      for (IDX s = tid; s < 2 * target_length + 1; s += blockDim.x) {
        beta_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, s)] = neginf;
      }
      if (tid == 0) {
        beta_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length)] =
            log_probs_ptr[input_helper.NdIndexToOffset(input_length - 1, b, blank)];
        if (target_length > 0) {
          int target =
              get_target_prime(targets_ptr, max_target_length, b, 2 * target_length - 1, blank);
          beta_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length - 1)] =
              log_probs_ptr[input_helper.NdIndexToOffset(input_length - 1, b, target)];
        }
      }
      __syncthreads();
    }
    for (IDX t = input_length - 2; t >= 0; t--) {
      for (IDX s = tid; s < 2 * target_length + 1; s += blockDim.x) {
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
            log(exp(lb1 - lbmax) + exp(lb2 - lbmax) + exp(lb3 - lbmax)) + lbmax
            + log_probs_ptr[input_helper.NdIndexToOffset(t, b, current_target_prime)];
      }
      __syncthreads();
    }
    for (IDX t = tid; t < max_input_length; t += blockDim.x) {
      for (IDX c = 0; c < num_labels; c++) {
        grad_ptr[input_helper.NdIndexToOffset(t, b, c)] = t < input_length ? neginf : 0;
      }
    }
    __syncthreads();
    if (tid == 0) {
      grad_ptr[input_helper.NdIndexToOffset(input_length - 1, b, blank)] =
          alpha_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length)]
          + beta_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length)];
      if (target_length > 0) {
        int target =
            get_target_prime(targets_ptr, max_target_length, b, 2 * target_length - 1, blank);
        grad_ptr[input_helper.NdIndexToOffset(input_length - 1, b, target)] =
            alpha_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length - 1)]
            + beta_ptr[beta_helper.NdIndexToOffset(b, input_length - 1, 2 * target_length - 1)];
      }
    }
    __syncthreads();
    for (IDX t = tid; t < input_length; t += blockDim.x) {
      for (IDX s = 0; (t < input_length - 1) && (s < 2 * target_length + 1); s += 1) {
        int current_target_prime = get_target_prime(targets_ptr, max_target_length, b, s, blank);
        int64_t idx_t_s = beta_helper.NdIndexToOffset(b, t, s);
        T log_alpha_beta = alpha_ptr[idx_t_s] + beta_ptr[idx_t_s];
        T& lcab = grad_ptr[input_helper.NdIndexToOffset(t, b, current_target_prime)];
        if (lcab == neginf) {
          lcab = log_alpha_beta;
        } else {
          T m = max(lcab, log_alpha_beta);
          lcab = log(exp(lcab - m) + exp(log_alpha_beta - m)) + m;
        }
      }
      for (int32_t c = 0; c < num_labels; c++) {
        T& res = grad_ptr[input_helper.NdIndexToOffset(t, b, c)];
        T lp = log_probs_ptr[input_helper.NdIndexToOffset(t, b, c)];
        res = (exp(lp) - exp(res + nll - lp)) * grad_out_ptr[b];
      }
    }
  }
}

}  // namespace

template<typename T, typename IDX>
struct CtcLossKernelUtil<DeviceType::kGPU, T, IDX> {
  static void CtcLossForward(DeviceCtx* ctx, const T* log_probs_ptr, const int* targets_ptr,
                             const IDX* input_lengths_ptr, const IDX* target_lengths_ptr,
                             T* alpha_ptr, T* loss_ptr,
                             NdIndexOffsetHelper<int64_t, 3>& input_helper,
                             NdIndexOffsetHelper<int64_t, 3>& alpha_helper,
                             const int64_t batch_size, const int64_t max_input_length,
                             const int64_t max_target_length, const int blank) {
    int32_t thread_num = batch_size * kCudaThreadsNumPerBlock;
    RUN_CUDA_KERNEL((CtcLossGpu<T, IDX>), ctx, thread_num, log_probs_ptr, targets_ptr,
                    input_lengths_ptr, target_lengths_ptr, alpha_ptr, loss_ptr, input_helper,
                    alpha_helper, batch_size, max_input_length, max_target_length, blank);
  }

  static void CtcLossBackward(DeviceCtx* ctx, const T* grad_out_ptr, const T* loss_ptr,
                              const T* alpha_ptr, const T* log_probs_ptr, const int* targets_ptr,
                              const IDX* input_lengths_ptr, const IDX* target_lengths_ptr,
                              T* beta_ptr, T* grad_ptr,
                              NdIndexOffsetHelper<int64_t, 3>& input_helper,
                              NdIndexOffsetHelper<int64_t, 3>& beta_helper,
                              const int64_t batch_size, const int64_t max_input_length,
                              const int64_t max_target_length, const int64_t num_labels,
                              const int blank, const bool zero_infinity) {
    int32_t thread_num = batch_size * kCudaThreadsNumPerBlock;
    RUN_CUDA_KERNEL((CtcLossGradGpu<T, IDX>), ctx, thread_num, grad_out_ptr, loss_ptr, alpha_ptr,
                    log_probs_ptr, targets_ptr, input_lengths_ptr, target_lengths_ptr, beta_ptr,
                    grad_ptr, input_helper, beta_helper, batch_size, max_input_length,
                    max_target_length, num_labels, blank, zero_infinity);
  }
};

#define INSTANTIATE_CTC_LOSS_KERNEL_UTIL_GPU(device_type_v, log_probs_dtype_pair,          \
                                             input_lengths_dtype_pair)                     \
  template struct CtcLossKernelUtil<device_type_v, OF_PP_PAIR_FIRST(log_probs_dtype_pair), \
                                    OF_PP_PAIR_FIRST(input_lengths_dtype_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CTC_LOSS_KERNEL_UTIL_GPU, (DeviceType::kGPU),
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef INSTANTIATE_CTC_LOSS_KERNEL_UTIL_GPU

}  // namespace oneflow
