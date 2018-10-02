#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/resize_nearest_neighbor_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"

namespace oneflow {

namespace {

template<typename T>
__global__ void ResizeNearestNeighborForward(const int64_t nthreads, const T* in_dptr,
                                             const int64_t channel_num, const int64_t height,
                                             const int64_t width, const int64_t new_height,
                                             const int64_t new_width, const float scale_h,
                                             const float scale_w, const bool align_corners,
                                             T* out_dptr) {
  const int64_t new_area = new_height * new_width;
  const int64_t channel_area = channel_num * height * width;
  const int64_t channel_new_area = channel_num * new_height * new_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / new_width) % new_height;
    const int64_t w = index % new_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             width - 1);
    out_dptr[index] = in_dptr[n * channel_area + (c * height + in_h) * width + in_w];
  }
}

template<typename T>
__global__ void ResizeNearestNeighborBackward(const int64_t nthreads, const T* out_diff_dptr,
                                              const int64_t channel_num, const int64_t height,
                                              const int64_t width, const int64_t new_height,
                                              const int64_t new_width, const float scale_h,
                                              const float scale_w, const bool align_corners,
                                              T* in_diff_dptr) {
  const int64_t area = height * width;
  const int64_t new_area = new_height * new_width;
  const int64_t channel_area = channel_num * height * width;
  const int64_t channel_new_area = channel_num * new_height * new_width;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    const int64_t h = (index / new_width) % new_height;
    const int64_t w = index % new_width;
    const int64_t c = (index / new_area) % channel_num;
    const int64_t n = index / channel_new_area;
    const int64_t in_h = min((align_corners) ? static_cast<int64_t>(roundf(h * scale_h))
                                             : static_cast<int64_t>(floorf(h * scale_h)),
                             height - 1);
    const int64_t in_w = min((align_corners) ? static_cast<int64_t>(roundf(w * scale_w))
                                             : static_cast<int64_t>(floorf(w * scale_w)),
                             width - 1);
    gpu_atomic_add(in_diff_dptr + n * channel_area + (c * height + in_h) * width + in_w,
                   out_diff_dptr[index]);
  }
}

}  // namespace

template<typename T>
struct ResizeNearestNeighborUtil<DeviceType::kGPU, T> {
  static void Forward(const KernelCtx& ctx, const ResizeNearestNeighborKernelConf& kernel_conf,
                      const bool align_corners, const Blob* in_blob, Blob* out_blob) {
    const int64_t elem_cnt = out_blob->shape().elem_cnt();

    ResizeNearestNeighborForward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                      ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, in_blob->dptr<T>(), in_blob->shape().At(1), in_blob->shape().At(2),
        in_blob->shape().At(3), out_blob->shape().At(2), out_blob->shape().At(3),
        kernel_conf.scale_h(), kernel_conf.scale_w(), align_corners, out_blob->mut_dptr<T>());
  }

  static void Backward(const KernelCtx& ctx, const ResizeNearestNeighborKernelConf& kernel_conf,
                       const bool align_corners, const Blob* out_diff_blob, Blob* in_diff_blob) {
    const int64_t elem_cnt = out_diff_blob->shape().elem_cnt();
    ResizeNearestNeighborBackward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                       ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, out_diff_blob->dptr<T>(), in_diff_blob->shape().At(1),
        in_diff_blob->shape().At(2), in_diff_blob->shape().At(3), out_diff_blob->shape().At(2),
        out_diff_blob->shape().At(3), kernel_conf.scale_h(), kernel_conf.scale_w(), align_corners,
        in_diff_blob->mut_dptr<T>());
  }
};

#define INSTANTIATE_RESIZE_NEAREST_NEIGHBOR_KERNEL_UTIL(type_cpp, type_proto) \
  template class ResizeNearestNeighborUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_RESIZE_NEAREST_NEIGHBOR_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
