#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/prelu_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/device/cuda_util.h"
#include <cub/cub.cuh>

namespace oneflow {
namespace {
template<typename T>
__global__ void PReluForward(const int64_t elem_cnt, const T* in_dptr, const T* alpha_dptr,
                             T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    out_dptr[i] = (in_dptr[i] >= 0) ? in_dptr[i] : in_dptr[i] * alpha_dptr[0];
  }
}

template<typename T>
__global__ void PReluForwardNCHW(const int64_t elem_cnt, const int64_t channel_num,
                                 const int64_t area, const T* in_dptr, const T* alpha_dptr,
                                 T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t c = (i / area) % channel_num;
    out_dptr[i] = (in_dptr[i] >= 0) ? in_dptr[i] : in_dptr[i] * alpha_dptr[c];
  }
}

template<typename T>
__global__ void PReluForwardNHWC(const int64_t elem_cnt, const int64_t channel_num,
                                 const T* in_dptr, const T* alpha_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t c = i % channel_num;
    out_dptr[i] = (in_dptr[i] >= 0) ? in_dptr[i] : in_dptr[i] * alpha_dptr[c];
  }
}

template<typename T>
__global__ void PReluSharedAlphaBackwardNCHW(const int64_t elem_cnt, const T* in_dptr,
                                             const T* out_diff_dptr, T* alpha_diff_dptr) {
  T alpha_sum = 0.0;
  for (int64_t i = threadIdx.x; i < elem_cnt; i += blockDim.x) {
    alpha_sum += (in_dptr[i] <= 0) ? out_diff_dptr[i] * in_dptr[i] : 0;
  }

  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T sum = BlockReduce(temp_storage).Sum(alpha_sum);
  if (threadIdx.x == 0) { *alpha_diff_dptr = sum; }
}

template<typename T>
__global__ void PReluDataBackward(const int64_t elem_cnt, const T* in_dptr, const T* alpha_dptr,
                                  const T* out_dff_dptr, T* in_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    in_diff_dptr[i] = (in_dptr[i] > 0) ? out_dff_dptr[i] : out_dff_dptr[i] * alpha_dptr[0];
  }
}

template<typename T>
__global__ void PReluAlphaBackwardNCHW(const int64_t channel_num, const int64_t instance_num,
                                       const int64_t area, const int64_t elem_cnt, const T* in_dptr,
                                       const T* out_diff_dptr, T* alpha_diff_dptr) {
  int64_t c = blockIdx.x;

  T alpha_sum = 0.0;
  int64_t channel_elem_cnt = elem_cnt / channel_num;
  for (int64_t i = threadIdx.x; i < channel_elem_cnt; i += blockDim.x) {
    int64_t n = i / area;
    int64_t ii = n * area * channel_num + c * area + i % area;
    alpha_sum += (in_dptr[ii] <= 0) ? out_diff_dptr[ii] * in_dptr[ii] : 0;
  }

  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T sum = BlockReduce(temp_storage).Sum(alpha_sum);
  if (threadIdx.x == 0) { alpha_diff_dptr[c] = sum; }
}

template<typename T>
__global__ void PReluDataBackwardNCHW(const int64_t elem_cnt, const int64_t channel_num,
                                      const int64_t area, const T* in_dptr, const T* alpha_dptr,
                                      const T* out_dff_dptr, T* in_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt * channel_num * area) {
    int64_t c = (i / area) % channel_num;
    in_diff_dptr[i] = (in_dptr[i] > 0) ? out_dff_dptr[i] : out_dff_dptr[i] * alpha_dptr[c];
  }
}

template<typename T>
__global__ void PReluAlphaBackwardNHWC(const int64_t channel_num, const int64_t elem_cnt,
                                       const T* in_dptr, const T* out_diff_dptr,
                                       T* alpha_diff_dptr) {
  int64_t c = blockIdx.x;
  T alpha_sum = 0.0;
  int64_t channel_elem_cnt = elem_cnt / channel_num;
  for (int64_t i = threadIdx.x; i < channel_elem_cnt; i += blockDim.x) {
    int64_t ii = i * channel_num + c;
    alpha_sum += (in_dptr[ii] <= 0) ? out_diff_dptr[ii] * in_dptr[ii] : 0;
  }

  typedef cub::BlockReduce<T, kCudaThreadsNumPerBlock> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T sum = BlockReduce(temp_storage).Sum(alpha_sum);
  if (threadIdx.x == 0) { alpha_diff_dptr[c] = sum; }
}

template<typename T>
__global__ void PReluDataBackwardNHWC(const int64_t elem_cnt, const int64_t channel_num,
                                      const T* in_dptr, const T* alpha_dptr, const T* out_dff_dptr,
                                      T* in_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t c = i % channel_num;
    in_diff_dptr[i] = (in_dptr[i] > 0) ? out_dff_dptr[i] : out_dff_dptr[i] * alpha_dptr[c];
  }
}

}  // namespace

template<typename T>
struct PReluKernelUtil<DeviceType::kGPU, T> {
  static void Forward(const KernelCtx& ctx, const PReluOpConf& conf, const Blob* in_blob,
                      const Blob* alpha_blob, Blob* out_blob) {
    const int64_t elem_cnt = in_blob->shape().elem_cnt();
    if (conf.channel_shared()) {
      PReluForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                     ctx.device_ctx->cuda_stream()>>>(
          elem_cnt, in_blob->dptr<T>(), alpha_blob->dptr<T>(), out_blob->mut_dptr<T>());
    } else {
      if (conf.data_format() == "channels_first") {
        const int64_t channel_num = in_blob->shape().At(1);
        const int64_t area = in_blob->shape().Count(2);
        PReluForwardNCHW<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                           ctx.device_ctx->cuda_stream()>>>(
            elem_cnt, channel_num, area, in_blob->dptr<T>(), alpha_blob->dptr<T>(),
            out_blob->mut_dptr<T>());
      } else if (conf.data_format() == "channels_last") {
        const int64_t channel_num = in_blob->shape().At(in_blob->shape().NumAxes() - 1);
        PReluForwardNHWC<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                           ctx.device_ctx->cuda_stream()>>>(
            elem_cnt, channel_num, in_blob->dptr<T>(), alpha_blob->dptr<T>(),
            out_blob->mut_dptr<T>());
      } else {
        UNIMPLEMENTED();
      }
    }
  }

  static void Backward(const KernelCtx& ctx, const PReluOpConf& conf, const Blob* in_blob,
                       const Blob* alpha_blob, const Blob* out_diff_blob, Blob* in_diff_blob,
                       Blob* alpha_diff_blob) {
    const int64_t elem_cnt = out_diff_blob->shape().elem_cnt();
    if (conf.channel_shared()) {
      PReluSharedAlphaBackwardNCHW<<<1, kCudaThreadsNumPerBlock, 0,
                                     ctx.device_ctx->cuda_stream()>>>(
          elem_cnt, in_blob->dptr<T>(), out_diff_blob->dptr<T>(), alpha_diff_blob->mut_dptr<T>());
      PReluDataBackward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(
          elem_cnt, in_blob->dptr<T>(), alpha_blob->dptr<T>(), out_diff_blob->dptr<T>(),
          in_diff_blob->mut_dptr<T>());
    } else {
      if (conf.data_format() == "channels_first") {
        const int64_t channel_num = out_diff_blob->shape().At(1);
        const int64_t instance_num = out_diff_blob->shape().At(0);
        const int64_t area = out_diff_blob->shape().Count(2);
        PReluAlphaBackwardNCHW<<<channel_num, kCudaThreadsNumPerBlock, 0,
                                 ctx.device_ctx->cuda_stream()>>>(
            channel_num, instance_num, area, elem_cnt, in_blob->dptr<T>(), out_diff_blob->dptr<T>(),
            alpha_diff_blob->mut_dptr<T>());
        PReluDataBackwardNCHW<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx.device_ctx->cuda_stream()>>>(
            instance_num, channel_num, area, in_blob->dptr<T>(), alpha_blob->dptr<T>(),
            out_diff_blob->dptr<T>(), in_diff_blob->mut_dptr<T>());
      } else if (conf.data_format() == "channels_last") {
        const int64_t channel_num = out_diff_blob->shape().At(in_blob->shape().NumAxes() - 1);
        PReluAlphaBackwardNHWC<<<channel_num, kCudaThreadsNumPerBlock, 0,
                                 ctx.device_ctx->cuda_stream()>>>(
            channel_num, elem_cnt, in_blob->dptr<T>(), out_diff_blob->dptr<T>(),
            alpha_diff_blob->mut_dptr<T>());
        PReluDataBackwardNHWC<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                ctx.device_ctx->cuda_stream()>>>(
            elem_cnt, channel_num, in_blob->dptr<T>(), alpha_blob->dptr<T>(),
            out_diff_blob->dptr<T>(), in_diff_blob->mut_dptr<T>());
      } else {
        UNIMPLEMENTED();
      }
    }
  }
};

#define INSTANTIATE_P_RELU_KERNEL_UTIL(type_cpp, type_proto) \
  template class PReluKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_P_RELU_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
