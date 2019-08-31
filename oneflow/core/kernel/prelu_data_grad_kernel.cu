#include "oneflow/core/kernel/prelu_data_grad_kernel.h"

namespace oneflow {
namespace {

template<typename T>
__global__ void PReluDataBackward(const int64_t elem_cnt, const int64_t channel_num,
                                  const int64_t area, const T* x, const T* alpha_dptr, const T* dy,
                                  T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t c = (i / area) % channel_num;
    dx[i] = (x[i] <= 0) ? dy[i] * alpha_dptr[c] : dy[i];
  }
}

}  // namespace

template<typename T>
struct PReluDataGradKernelUtil<DeviceType::kGPU, T> {
  static void Compute(const KernelCtx& ctx, const PReluDataGradOpConf& conf, const Blob* x_blob,
                      const Blob* alpha_blob, const Blob* dy_blob, Blob* dx_blob) {
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    if (conf.channel_shared()) {
      PReluDataBackward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx.device_ctx->cuda_stream()>>>(
          elem_cnt, 1, 1, x_blob->dptr<T>(), alpha_blob->dptr<T>(), dy_blob->dptr<T>(),
          dx_blob->mut_dptr<T>());
    } else {
      if (conf.data_format() == "channels_first") {
        const int64_t channel_num = dy_blob->shape().At(1);
        PReluDataBackward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                            ctx.device_ctx->cuda_stream()>>>(
            elem_cnt, channel_num, dy_blob->shape().Count(2), x_blob->dptr<T>(),
            alpha_blob->dptr<T>(), dy_blob->dptr<T>(), dx_blob->mut_dptr<T>());
      } else if (conf.data_format() == "channels_last") {
        const int64_t channel_num = dy_blob->shape().At(x_blob->shape().NumAxes() - 1);
        PReluDataBackward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                            ctx.device_ctx->cuda_stream()>>>(
            elem_cnt, channel_num, 1, x_blob->dptr<T>(), alpha_blob->dptr<T>(), dy_blob->dptr<T>(),
            dx_blob->mut_dptr<T>());
      } else {
        UNIMPLEMENTED();
      }
    }
  }
};

#define INSTANTIATE_P_RELU_DATA_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template class PReluDataGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_P_RELU_DATA_GRAD_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
