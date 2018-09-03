#include "oneflow/core/kernel/smooth_l1_loss_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SmoothL1LossForward(const int64_t instance_num, const int64_t instance_dim,
                                    const T* prediction, const T* label, const T* inside_weights,
                                    const T* outside_weights, const float beta, const float scale,
                                    T* loss) {
  int64_t elem_cnt = instance_num * instance_dim;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    T x = inside_weights[i] * (prediction[i] - label[i]);
    T abs_x = std::abs(x);
    if (abs_x < beta) {
      loss[i] = 0.5 * x * x / beta;
    } else {
      loss[i] = abs_x - 0.5 * beta;
    }
    loss[i] *= scale / instance_num * outside_weights[i];
  }
}

template<typename T>
__global__ void SmoothL1LossBackward(const int64_t instance_num, const int64_t instance_dim,
                                     const T* prediction, const T* label, const T* inside_weights,
                                     const T* outside_weights, const float beta, const float scale,
                                     T* in_diff) {
  int64_t elem_cnt = instance_num * instance_dim;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    T x = inside_weights[i] * (prediction[i] - label[i]);
    T abs_x = std::abs(x);
    if (abs_x < beta) {
      in_diff[i] = x / beta;
    } else {
      in_diff[i] = (x > ZeroVal<T>::value) - (x < ZeroVal<T>::value);
    }
    in_diff[i] *= scale / instance_num * inside_weights[i] * outside_weights[i];
  }
}

}  // namespace

template<typename T>
struct SmoothL1LossKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const int64_t instance_num, const int64_t instance_dim,
                      const T* prediction, const T* label, const T* inside_weights,
                      const T* outside_weights, const float beta, const float scale, T* loss) {
    SmoothL1LossForward<T>
        <<<BlocksNum4ThreadsNum(instance_num * instance_dim), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(instance_num, instance_dim, prediction, label, inside_weights,
                                 outside_weights, beta, scale, loss);
  }
  static void Backward(DeviceCtx* ctx, const int64_t instance_num, const int64_t instance_dim,
                       const T* prediction, const T* label, const T* inside_weights,
                       const T* outside_weights, const float beta, const float scale, T* in_diff) {
    SmoothL1LossBackward<T>
        <<<BlocksNum4ThreadsNum(instance_num * instance_dim), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(instance_num, instance_dim, prediction, label, inside_weights,
                                 outside_weights, beta, scale, in_diff);
  }
};

#define MAKE_ENTRY(data_type_cpp, data_type_proto) \
  template struct SmoothL1LossKernelUtil<DeviceType::kGPU, data_type_cpp>;
OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)
}  // namespace oneflow
