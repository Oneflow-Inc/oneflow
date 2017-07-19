#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

namespace {

template<typename FloatingPointType>
__global__ void RangeMaxQueryGpuKernel(
    const FloatingPointType* in_dptr, const int64_t in_height,
    const int64_t in_width, const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend, const int64_t pool_size,
    const int64_t out_index, FloatingPointType* out_dptr, int64_t* mask_dptr) {
  out_dptr[out_index] = in_dptr[hstart * in_width + wstart];
  mask_dptr[out_index] = hstart * in_width + wstart;
  for (int64_t h = hstart; h < hend; ++h) {
    for (int64_t w = wstart; w < wend; ++w) {
      const int64_t index = h * in_width + w;
      if (in_dptr[index] > out_dptr[out_index]) {
        out_dptr[out_index] = in_dptr[index];
        mask_dptr[out_index] = index;
      }
    }
  }
}

template<typename FloatingPointType>
__global__ void RangeAveQueryGpuKernel(
    const FloatingPointType* in_dptr, const int64_t in_height,
    const int64_t in_width, const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend, const int64_t pool_size,
    const int64_t out_index, FloatingPointType* out_dptr, int64_t* mask_dptr) {
  out_dptr[out_index] = 0;
  for (int64_t h = hstart; h < hend; ++h) {
    for (int64_t w = wstart; w < wend; ++w) {
      const int64_t index = h * in_width + w;
      out_dptr[out_index] += in_dptr[index];
    }
  }
  out_dptr[out_index] /= pool_size;
}

// TODO(shiyuan) random function
template<typename FloatingPointType>
__global__ void RangeStoQueryGpuKernel(
    const FloatingPointType* in_dptr, const int64_t in_height,
    const int64_t in_width, const int64_t hstart, const int64_t wstart,
    const int64_t hend, const int64_t wend, const int64_t pool_size,
    const int64_t out_index, FloatingPointType* out_dptr, int64_t* mask_dptr) {
  const int64_t index =
      (hstart /*+ random()*/) * in_width + (wstart /*+ random()*/);
  out_dptr[out_index] = in_dptr[index];
  mask_dptr[out_index] = index;
}

template<typename FloatingPointType>
__global__ void PoolingMaxBpGpuKernel(
    const FloatingPointType* out_diff_dptr, const int64_t* mask_dptr,
    const int64_t pool_size, const int64_t out_diff_index,
    const int64_t in_height, const int64_t in_width, const int64_t hstart,
    const int64_t wstart, const int64_t hend, const int64_t wend,
    FloatingPointType* in_diff_dptr) {
  const int64_t in_diff_index = mask_dptr[out_diff_index];
  in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
}

template<typename FloatingPointType>
__global__ void PoolingAveBpGpuKernel(
    const FloatingPointType* out_diff_dptr, const int64_t* mask_dptr,
    const int64_t pool_size, const int64_t out_diff_index,
    const int64_t in_height, const int64_t in_width, const int64_t hstart,
    const int64_t wstart, const int64_t hend, const int64_t wend,
    FloatingPointType* in_diff_dptr) {
  for (int h = hstart; h < hend; ++h) {
    for (int w = wstart; w < wend; ++w) {
      in_diff_dptr[h * in_width + w] += out_diff_dptr[out_diff_index] / pool_size;
    }
  }
}

template<typename FloatingPointType>
__global__ void PoolingStoBpGpuKernel(
    const FloatingPointType* out_diff_dptr, const int64_t* mask_dptr,
    const int64_t pool_size, const int64_t out_diff_index,
    const int64_t in_height, const int64_t in_width, const int64_t hstart,
    const int64_t wstart, const int64_t hend, const int64_t wend,
    FloatingPointType* in_diff_dptr) {
  const int64_t in_diff_index = mask_dptr[out_diff_index];
  in_diff_dptr[in_diff_index] += out_diff_dptr[out_diff_index];
}

}  // namespace

template<typename FloatingPointType>
class PoolingKernelUtil<DeviceType::kGPU, FloatingPointType> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingKernelUtil);
  PoolingKernelUtil() = delete;

  static void RangeMaxQuery(const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr) {
    RangeMaxQueryGpuKernel<FloatingPointType>
        <<<BlocksNum4ThreadsNum(1), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            in_dptr, in_height, in_width, hstart, wstart, hend, wend, pool_size,
            out_index, out_dptr, mask_dptr);
  }

  static void RangeAveQuery(const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr) {
    RangeAveQueryGpuKernel<FloatingPointType>
        <<<BlocksNum4ThreadsNum(1), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            in_dptr, in_height, in_width, hstart, wstart, hend, wend, pool_size,
            out_index, out_dptr, mask_dptr);
  }

  static void RangeStoQuery(const FloatingPointType* in_dptr,
                            const int64_t in_height, const int64_t in_width,
                            const int64_t hstart, const int64_t wstart,
                            const int64_t hend, const int64_t wend,
                            const int64_t pool_size, const int64_t out_index,
                            FloatingPointType* out_dptr, int64_t* mask_dptr) {
    RangeStoQueryGpuKernel<FloatingPointType>
        <<<BlocksNum4ThreadsNum(1), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            in_dptr, in_height, in_width, hstart, wstart, hend, wend, pool_size,
            out_index, out_dptr, mask_dptr);
  }

  static void PoolingMaxBp(const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr) {
    PoolingMaxBpGpuKernel<FloatingPointType>
        <<<BlocksNum4ThreadsNum(1), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            out_diff_dptr, mask_dptr, pool_size, out_diff_index, in_height,
            in_width, hstart, wstart, hend, wend, in_diff_dptr);
  }

  static void PoolingAveBp(const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr) {
    PoolingAveBpGpuKernel<FloatingPointType>
        <<<BlocksNum4ThreadsNum(1), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            out_diff_dptr, mask_dptr, pool_size, out_diff_index, in_height,
            in_width, hstart, wstart, hend, wend, in_diff_dptr);
  }

  static void PoolingStoBp(const FloatingPointType* out_diff_dptr,
                           const int64_t* mask_dptr, const int64_t pool_size,
                           const int64_t out_diff_index,
                           const int64_t in_height, const int64_t in_width,
                           const int64_t hstart, const int64_t wstart,
                           const int64_t hend, const int64_t wend,
                           FloatingPointType* in_diff_dptr) {
    PoolingStoBpGpuKernel<FloatingPointType>
        <<<BlocksNum4ThreadsNum(1), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(
            out_diff_dptr, mask_dptr, pool_size, out_diff_index, in_height,
            in_width, hstart, wstart, hend, wend, in_diff_dptr);
  }
};

INSTANTIATE_GPU_KERNEL_UTIL_CLASS(PoolingKernelUtil);

}  // namespace oneflow
