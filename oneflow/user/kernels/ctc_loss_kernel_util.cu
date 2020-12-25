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
__global__ void CtcLossGpu(const IDX batch_size, const T* log_probs_ptr, const int* targets_ptr,
                           const IDX* input_lengths_ptr, const IDX* target_length_ptr, T* alpha_ptr,
                           T* loss_ptr, NdIndexOffsetHelper<IDX, 3> input_helper,
                           NdIndexOffsetHelper<IDX, 3> alpha_helper, IDX max_target_length,
                           const int blank) {
  constexpr T neginf = -INFINITY;
  CUDA_1D_KERNEL_LOOP(b, batch_size) {
    IDX input_length = input_lengths_ptr[b];
    IDX target_length = target_length_ptr[b];

    IDX alpha_index = alpha_helper.NdIndexToOffset(b, 0, 0);
    for (IDX s = 0; s < 2 * target_length + 1; s++) { alpha_ptr[alpha_index + s] = neginf; }

    alpha_ptr[alpha_index] = log_probs_ptr[input_helper.NdIndexToOffset(0, b, blank)];
    if (target_length > 0) {
      int target = get_target_prime(targets_ptr, max_target_length, b, 1, blank);
      alpha_ptr[alpha_index + 1] = log_probs_ptr[input_helper.NdIndexToOffset(0, b, target)];
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

        IDX idx_t_s = alpha_helper.NdIndexToOffset(b, t, s);
        alpha_ptr[idx_t_s] =
            std::log(std::exp(la1 - lamax) + std::exp(la2 - lamax) + std::exp(la3 - lamax)) + lamax
            + log_probs_ptr[input_helper.NdIndexToOffset(t, b, current_target_prime)];
      }
    }

    if (target_length == 0) {
      IDX idx = alpha_helper.NdIndexToOffset(b, input_length - 1, 0);
      loss_ptr[b] = -alpha_ptr[idx];
    } else {
      IDX idx1 = alpha_helper.NdIndexToOffset(b, input_length - 1, target_length * 2);
      IDX idx2 = alpha_helper.NdIndexToOffset(b, input_length - 1, target_length * 2 - 1);
      T l1 = alpha_ptr[idx1];
      T l2 = alpha_ptr[idx2];
      T m = max(l1, l2);
      m = ((m == neginf) ? 0 : m);
      T log_likelihood = log(exp(l1 - m) + exp(l2 - m)) + m;
      loss_ptr[b] = -log_likelihood;
    }
  }
}

}  // namespace

template<typename T, typename IDX>
struct CtcLossKernelUtil<DeviceType::kGPU, T, IDX> {
  static void CtcLossForward(DeviceCtx* ctx, const IDX batch_size, const T* log_probs_ptr,
                             const int* targets_ptr, const IDX* input_lengths_ptr,
                             const IDX* target_length_ptr, T* alpha_ptr, T* loss_ptr,
                             NdIndexOffsetHelper<IDX, 3> input_helper,
                             NdIndexOffsetHelper<IDX, 3> alpha_helper, IDX max_target_length,
                             const int blank) {
    RUN_CUDA_KERNEL((CtcLossGpu<T, IDX>), ctx, batch_size, batch_size, log_probs_ptr, targets_ptr,
                    input_lengths_ptr, target_length_ptr, alpha_ptr, loss_ptr, input_helper,
                    alpha_helper, max_target_length, blank);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_CTC_LOSS_FUNCTOR, (DeviceType::kGPU),
                                 OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat), INDEX_DATA_TYPE_SEQ)

}  // namespace oneflow
