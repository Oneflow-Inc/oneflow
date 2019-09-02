#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/adam_model_update_kernel.h"
#include "oneflow/core/kernel/normal_model_update_kernel.cuh"

namespace oneflow {

namespace {

template<int32_t power>
struct PowUtil;

template<>
struct PowUtil<1> final {
  template<typename T>
  __device__ static T pow(const T x) {
    return x;
  }
};

template<>
struct PowUtil<2> final {
  template<typename T>
  __device__ static T pow(const T x) {
    return x * x;
  }
};

template<bool do_bias_correction, typename T>
__device__ typename std::enable_if<do_bias_correction>::type ScaleMomentum(const T beta_t,
                                                                           T* moment) {
  *moment /= (1 - beta_t);
}

template<bool do_bias_correction, typename T>
__device__ typename std::enable_if<!do_bias_correction>::type ScaleMomentum(const T beta_t,
                                                                            T* moment) {}

template<int32_t power, bool do_bias_correction, typename T>
__device__ void UpdateMomentEstimate(T beta, const T* model_diff, const T* beta_t, T* moment) {
  // Update biased moment estimate
  *moment = beta * (*moment) + (1 - beta) * PowUtil<power>::pow(*model_diff);
  // Correct deviation of moment estimate
  ScaleMomentum<do_bias_correction>(*beta_t, moment);
}

template<typename T>
__device__ void UpdateModel(const float* learning_rate, T l1, T l2, T epsilon, T* model_diff,
                            T* model, T* m, T* v) {
  *model_diff = *m / (sqrt(*v) + epsilon);
  T reg_diff = RegDiff(*model_diff, l1, l2, *model);
  *model = *model - *learning_rate * reg_diff;
}

template<bool do_bias_correction, typename T>
__global__ void UpdateModelGpu(int64_t n, const float* learning_rate, T l1, T l2, T beta1, T beta2,
                               T epsilon, const T* beta1_t, const T* beta2_t, T* model_diff,
                               T* model, T* m, T* v) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    UpdateMomentEstimate<1, do_bias_correction>(beta1, model_diff + i, beta1_t, m + i);
    UpdateMomentEstimate<2, do_bias_correction>(beta2, model_diff + i, beta2_t, v + i);
    UpdateModel(learning_rate, l1, l2, epsilon, model_diff + i, model + i, m + i, v + i);
  }
}

template<typename T>
__global__ void DoBiasCorrectionGpu(const int64_t* global_step, const T beta1, const T beta2,
                                    T* beta1_t, T* beta2_t) {
  if (*global_step != 0) {
    *beta1_t *= beta1;
    *beta2_t *= beta2;
  }
}

}  // namespace

template<typename T>
class AdamMdUpdateKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void UpdateModel(DeviceCtx* ctx, int64_t n, const float* learning_rate, T l1, T l2,
                          T beta1, T beta2, T epsilon, bool do_bias_correction,
                          const int64_t* global_step, const T* beta1_t, const T* beta2_t,
                          T* model_diff, T* model, T* m, T* v) {
    if (do_bias_correction) {
      UpdateModelGpu<true, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              n, learning_rate, l1, l2, beta1, beta2, epsilon, beta1_t, beta2_t, model_diff, model,
              m, v);
    } else {
      UpdateModelGpu<false, T>
          <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
              n, learning_rate, l1, l2, beta1, beta2, epsilon, beta1_t, beta2_t, model_diff, model,
              m, v);
    }
  }

  static void DoBiasCorrection(DeviceCtx* ctx, const int64_t* global_step, const T beta1,
                               const T beta2, T* beta1_t, T* beta2_t) {
    DoBiasCorrectionGpu<T>
        <<<1, 1, 0, ctx->cuda_stream()>>>(global_step, beta1, beta2, beta1_t, beta2_t);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class AdamMdUpdateKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
