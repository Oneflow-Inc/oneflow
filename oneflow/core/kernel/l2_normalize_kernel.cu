#include "oneflow/core/kernel/l2_normalize_kernel.h"
#include <cub/cub.cuh>
#include <math.h>

namespace oneflow {

namespace {

template<typename T>
__global__ void L2NormalizeForward(const int32_t n, const int32_t c, const int32_t d,
                                   const T epsilon, const T* in, T* norm, T* out) {
  using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    T sum = ZeroVal<T>::value;
    int32_t beg = (i / d) * d * c + (i % d);
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const T x = in[beg + j * d];
      sum += x * x;
    }

    T reduce_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) { norm[i] = fmaxf(std::sqrt(reduce_sum), epsilon); }
    __syncthreads();

    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const int32_t index = beg + j * d;
      out[index] = in[index] / norm[i];
    }
  }
}

template<typename T>
__global__ void L2NormalizeBackward(const int32_t n, const int32_t c, const int32_t d, const T* in,
                                    const T* out_diff, const T* norm, T* in_diff) {
  using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage_prod_sum;

  for (int32_t i = blockIdx.x; i < n; i += gridDim.x) {
    T x_dy_prod_dum = ZeroVal<T>::value;
    int32_t beg = (i / d) * d * c + (i % d);
    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const int32_t index = beg + j * d;
      x_dy_prod_dum += in[index] * out_diff[index];
    }

    T reduce_x_dy_prod_sum = BlockReduce(temp_storage_prod_sum).Sum(x_dy_prod_dum);
    __shared__ T x_dy_inner_prod;
    if (threadIdx.x == 0) { x_dy_inner_prod = reduce_x_dy_prod_sum; }
    __syncthreads();

    for (int32_t j = threadIdx.x; j < c; j += blockDim.x) {
      const int32_t index = beg + j * d;
      in_diff[index] =
          (out_diff[index] / norm[i]) - ((in[index] / (std::pow(norm[i], 3))) * x_dy_inner_prod);
    }
  }
}

}  // namespace

template<typename T>
struct L2NormalizeKernelUtil<DeviceType::kGPU, T> {
  static void Forward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                      Blob* norm_blob, Blob* out_blob) {
    int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + in_blob->shape().NumAxes();
    int32_t c = in_blob->shape().At(axis);
    int32_t n = in_blob->shape().elem_cnt() / c;
    int32_t d = in_blob->shape().Count(axis + 1);
    L2NormalizeForward<<<std::min(n, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
                         ctx->cuda_stream()>>>(n, c, d, static_cast<T>(conf.epsilon()),
                                               in_blob->dptr<T>(), norm_blob->mut_dptr<T>(),
                                               out_blob->mut_dptr<T>());
  }

  static void Backward(DeviceCtx* ctx, const L2NormalizeOpConf& conf, const Blob* in_blob,
                       const Blob* out_diff_blob, const Blob* norm_blob, Blob* in_diff_blob) {
    int32_t axis = conf.axis() >= 0 ? conf.axis() : conf.axis() + in_blob->shape().NumAxes();
    int32_t c = in_blob->shape().At(axis);
    int32_t n = in_blob->shape().elem_cnt() / c;
    int32_t d = in_blob->shape().Count(axis + 1);
    L2NormalizeBackward<<<std::min(n, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
                          ctx->cuda_stream()>>>(n, c, d, in_blob->dptr<T>(),
                                                out_diff_blob->dptr<T>(), norm_blob->dptr<T>(),
                                                in_diff_blob->mut_dptr<T>());
  }
};

#define INSTANTIATE_L2_NORMALIZE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct L2NormalizeKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_L2_NORMALIZE_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
