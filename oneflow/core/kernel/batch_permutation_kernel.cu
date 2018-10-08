#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/batch_permutation_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {
template<typename T>
__global__ void BatchPermutationForward(int64_t elem_cnt, const T* in_dptr, const int32_t* indices,
                                        int64_t channel_num, int64_t height, int64_t width,
                                        T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t w = index % width;
    int64_t h = (index / width) % height;
    int64_t c = (index / height / width) % channel_num;
    int64_t n = (index / channel_num / height / width);
    int64_t idx = indices[n];
    out_dptr[n * channel_num * height * width + c * height * width + h * width + w] =
        in_dptr[idx * channel_num * height * width + c * height * width + h * width + w];
  }
}

template<typename T>
__global__ void BatchPermutationBackward(int64_t elem_cnt, const T* out_diff_dptr,
                                         const int32_t* indices, int64_t channel_num,
                                         int64_t height, int64_t width, T* in_diff_dptr) {
  CUDA_1D_KERNEL_LOOP(index, elem_cnt) {
    int64_t w = index % width;
    int64_t h = (index / width) % height;
    int64_t c = (index / height / width) % channel_num;
    int64_t n = (index / channel_num / height / width);
    int64_t idx = indices[n];
    in_diff_dptr[idx * channel_num * height * width + c * height * width + h * width + w] =
        out_diff_dptr[n * channel_num * height * width + c * height * width + h * width + w];
  }
}

}  // namespace

template<typename T>
struct BatchPermutationUtil<DeviceType::kGPU, T> {
  static void Forward(const KernelCtx& ctx, const BatchPermutationOpConf& conf, const Blob* in_blob,
                      const Blob* indices_blob, Blob* out_blob) {
    const int64_t elem_cnt = out_blob->shape().elem_cnt();
    BatchPermutationForward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                 ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, in_blob->dptr<T>(), indices_blob->dptr<int32_t>(), in_blob->shape().At(1),
        in_blob->shape().At(2), in_blob->shape().At(3), out_blob->mut_dptr<T>());
  }

  static void Backward(const KernelCtx& ctx, const BatchPermutationOpConf& conf,
                       const Blob* out_diff_blob, const Blob* indices_blob, Blob* in_diff_blob) {
    const int64_t elem_cnt = in_diff_blob->shape().elem_cnt();
    BatchPermutationBackward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                  ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, out_diff_blob->dptr<T>(), indices_blob->dptr<int32_t>(),
        out_diff_blob->shape().At(1), out_diff_blob->shape().At(2), out_diff_blob->shape().At(3),
        in_diff_blob->mut_dptr<T>());
  }
};

#define INSTANTIATE_BATCH_PERMUTATION_KERNEL_UTIL(type_cpp, type_proto) \
  template class BatchPermutationUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_BATCH_PERMUTATION_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
