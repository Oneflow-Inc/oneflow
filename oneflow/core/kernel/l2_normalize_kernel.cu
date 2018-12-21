#include "oneflow/core/kernel/l2_normalize_kernel.h"
#include <cub/cub.cuh>
#include <math.h>

namespace oneflow {

namespace {

template<typename T>
__global__ void L2NormalizeForward(const int32_t n, const int32_t c, const int32_t d,
                                   const T epsilon, const T* in, T* square_x_sum, T* out) {
  using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    T sum = ZeroVal<T>::value;
    const int32_t offset = (i / d) * d * c + (i % d);
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const T x = in[offset + j * d];
      sum += x * x;
    }
    const T reduce_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) { square_x_sum[i] = reduce_sum; }
    __syncthreads();

    const T inv_norm = rsqrtf(fmaxf(square_x_sum[i], epsilon));
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const int32_t index = offset + j * d;
      out[index] = inv_norm * in[index];
    }
  }
}

template<typename T>
__global__ void L2NormalizeBackward(const int32_t n, const int32_t c, const int32_t d,
                                    const float epsilon, const T* out, const T* out_diff,
                                    const T* square_x_sum, T* in_diff) {
  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    const T inv_norm = rsqrt(fmaxf(square_x_sum[i], epsilon));
    const int32_t offset = (i / d) * d * c + (i % d);
    if (square_x_sum[i] >= epsilon) {
      using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
      __shared__ typename BlockReduce::TempStorage temp_storage_prod_sum;

      T y_dy_prod_sum = ZeroVal<T>::value;
      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        y_dy_prod_sum += out[index] * out_diff[index];
      }

      const T reduce_y_dy_prod_sum = BlockReduce(temp_storage_prod_sum).Sum(y_dy_prod_sum);
      __shared__ T y_dy_inner_prod;
      if (threadIdx.x == 0) { y_dy_inner_prod = reduce_y_dy_prod_sum; }
      __syncthreads();

      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        in_diff[index] = inv_norm * (out_diff[index] - y_dy_inner_prod * out[index]);
      }
    } else {
      for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t index = offset + j * d;
        in_diff[index] = inv_norm * out_diff[index];
      }
    }
  }
}

}  // namespace

template<typename T>
struct L2NormalizeKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                      Blob* square_x_sum_blob, Blob* out_blob) {
    int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + in_blob->shape().NumAxes();
    int32_t c = in_blob->shape().At(axis);
    int32_t n = in_blob->shape().elem_cnt() / c;
    int32_t d = in_blob->shape().Count(axis + 1);
    L2NormalizeForward<<<std::min(n, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
                         ctx->cuda_stream()>>>(n, c, d, static_cast<T>(conf.epsilon()),
                                               in_blob->dptr<T>(), square_x_sum_blob->mut_dptr<T>(),
                                               out_blob->mut_dptr<T>());
  }

  static void Backward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* out_blob,
                       const Blob* out_diff_blob, const Blob* square_x_sum_blob,
                       Blob* in_diff_blob) {
    int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + out_blob->shape().NumAxes();
    int32_t c = out_blob->shape().At(axis);
    int32_t n = out_blob->shape().elem_cnt() / c;
    int32_t d = out_blob->shape().Count(axis + 1);
    L2NormalizeBackward<<<std::min(n, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
                          ctx->cuda_stream()>>>(
        n, c, d, static_cast<T>(conf.epsilon()), out_blob->dptr<T>(), out_diff_blob->dptr<T>(),
        square_x_sum_blob->dptr<T>(), in_diff_blob->mut_dptr<T>());
  }
};

#define INSTANTIATE_L2_NORMALIZE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct L2NormalizeKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_L2_NORMALIZE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
