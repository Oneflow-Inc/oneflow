#include "oneflow/core/kernel/affine_channel_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/transpose_kernel.h"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T>
__global__ void ForwardGpu(const int64_t elem_cnt, const int32_t channel_dim,
                           const int64_t channel_stride, const T* in, const T* scale, const T* bias,
                           T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const int32_t channel_i = (i / channel_stride) % channel_dim;
    if (bias != nullptr) {
      out[i] = in[i] * scale[channel_i] + bias[channel_i];
    } else {
      out[i] = in[i] * scale[channel_i];
    }
  }
}

template<typename T>
__global__ void BackwardInDiffGpu(const int64_t elem_cnt, const int32_t channel_dim,
                                  const int64_t channel_stride, const T* out_diff, const T* scale,
                                  T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    in_diff[i] = out_diff[i] * scale[(i / channel_stride) % channel_dim];
  }
}

template<typename T>
__global__ void BackwardScaleBiasDiffGpu(const int64_t elem_cnt, const int32_t channel_dim,
                                         const int64_t channel_stride, const T* in,
                                         const T* out_diff, T* scale_diff, T* bias_diff) {
  using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
  __shared__ typename BlockReduce::TempStorage scale_diff_temp_storage;
  __shared__ typename BlockReduce::TempStorage bias_diff_temp_storage;

  for (int32_t i = blockIdx.x; i < channel_dim; i += gridDim.x) {
    T scale_diff_sum = ZeroVal<T>::value;
    T bias_diff_sum = ZeroVal<T>::value;
    for (int64_t j = threadIdx.x; j < (elem_cnt / channel_dim); j += blockDim.x) {
      int64_t index =
          ((j / channel_stride) * channel_dim + i) * channel_stride + j % channel_stride;
      scale_diff_sum += out_diff[index] * in[index];
      bias_diff_sum += out_diff[index];
    }
    T reduce_scale_diff_sum = BlockReduce(scale_diff_temp_storage).Sum(scale_diff_sum);
    T reduce_bias_diff_sum = BlockReduce(bias_diff_temp_storage).Sum(bias_diff_sum);
    if (threadIdx.x == 0) {
      scale_diff[i] = reduce_scale_diff_sum;
      bias_diff[i] = reduce_bias_diff_sum;
    }
  }
}

template<typename T>
__global__ void BackwardScaleDiffGpu(const int64_t elem_cnt, const int32_t channel_dim,
                                     const int64_t channel_stride, const T* in, const T* out_diff,
                                     T* scale_diff) {
  using BlockReduce = cub::BlockReduce<T, kCudaThreadsNumPerBlock>;
  __shared__ typename BlockReduce::TempStorage scale_diff_temp_storage;

  for (int32_t i = blockIdx.x; i < channel_dim; i += gridDim.x) {
    T scale_diff_sum = ZeroVal<T>::value;
    for (int64_t j = threadIdx.x; j < (elem_cnt / channel_dim); j += blockDim.x) {
      int64_t index =
          ((j / channel_stride) * channel_dim + i) * channel_stride + j % channel_stride;
      scale_diff_sum += out_diff[index] * in[index];
    }
    T reduce_scale_diff_sum = BlockReduce(scale_diff_temp_storage).Sum(scale_diff_sum);
    if (threadIdx.x == 0) { scale_diff[i] = reduce_scale_diff_sum; }
  }
}

}  // namespace

template<typename T>
class AffineChannelKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void Forward(DeviceCtx* ctx, const int64_t elem_cnt, const int32_t channel_dim,
                      const int64_t channel_stride, const T* in, const T* scale, const T* bias,
                      T* out) {
    ForwardGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, channel_dim, channel_stride, in, scale, bias, out);
  }

  static void BackwardInDiff(DeviceCtx* ctx, const int64_t elem_cnt, const int32_t channel_dim,
                             const int64_t channel_stride, const T* out_diff, const T* scale,
                             T* in_diff) {
    BackwardInDiffGpu<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            elem_cnt, channel_dim, channel_stride, out_diff, scale, in_diff);
  }

  static void BackwardScaleBiasDiff(DeviceCtx* ctx, const int64_t elem_cnt,
                                    const int32_t channel_dim, const int64_t channel_stride,
                                    const T* in, const T* out_diff, T* scale_diff, T* bias_diff) {
    BackwardScaleBiasDiffGpu<T><<<std::min(channel_dim, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock,
                                  0, ctx->cuda_stream()>>>(elem_cnt, channel_dim, channel_stride,
                                                           in, out_diff, scale_diff, bias_diff);
  }

  static void BackwardScaleDiff(DeviceCtx* ctx, const int64_t elem_cnt, const int32_t channel_dim,
                                const int64_t channel_stride, const T* in, const T* out_diff,
                                T* scale_diff) {
    BackwardScaleDiffGpu<T>
        <<<std::min(channel_dim, kCudaMaxBlocksNum), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(elem_cnt, channel_dim, channel_stride, in, out_diff, scale_diff);
  }
};

#define INSTANTIATE_GPU_KERNEL_UTIL(type_cpp, type_proto) \
  template class AffineChannelKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
