#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/prelu_alpha_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {
namespace {

template<typename T>
__global__ void PReluAlphaBackward(const int64_t elem_cnt, const T* x, const T* dy,
                                   T* alpha_grad_buf_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { alpha_grad_buf_dptr[i] = (x[i] <= 0) ? dy[i] * x[i] : 0; }
}

}  // namespace

template<typename T>
struct PReluAlphaGradKernelUtil<DeviceType::kGPU, T> {
  static void Compute(const KernelCtx& ctx, const PReluAlphaGradOpConf& conf,
                      const PbRf<int32_t>& permutation, const Blob* x_blob, const Blob* dy_blob,
                      Blob* bw_buf_blob, Blob* alpha_grad_buf_blob, Blob* alpha_grad_blob) {
    const int64_t elem_cnt = dy_blob->shape().elem_cnt();
    PReluAlphaBackward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx.device_ctx->cuda_stream()>>>(
        elem_cnt, x_blob->dptr<T>(), dy_blob->dptr<T>(), alpha_grad_buf_blob->mut_dptr<T>());
    if (conf.channel_shared()) {
      KernelUtil<DeviceType::kGPU, T>::Sum(
          ctx.device_ctx, elem_cnt, alpha_grad_buf_blob->dptr<T>(), alpha_grad_blob->mut_dptr<T>(),
          bw_buf_blob->mut_dptr<T>(), bw_buf_blob->ByteSizeOfDataContentField());
    } else {
      KernelUtil<DeviceType::kGPU, T>::Transpose(
          ctx.device_ctx, alpha_grad_buf_blob->shape().NumAxes(), alpha_grad_buf_blob->shape(),
          bw_buf_blob->shape(), permutation, alpha_grad_buf_blob->shape().elem_cnt(),
          alpha_grad_buf_blob->dptr<T>(), bw_buf_blob->mut_dptr<T>());
      CHECK_EQ(elem_cnt, bw_buf_blob->shape().elem_cnt());
      if (conf.data_format() == "channels_first") {
        const int64_t channel_num = dy_blob->shape().At(1);
        CHECK_EQ(channel_num, bw_buf_blob->shape().At(0));
        KernelUtil<DeviceType::kGPU, T>::RowSum(
            ctx.device_ctx, channel_num, bw_buf_blob->shape().Count(1), bw_buf_blob->dptr<T>(),
            alpha_grad_blob->mut_dptr<T>(), alpha_grad_buf_blob->mut_dptr<T>(),
            alpha_grad_buf_blob->ByteSizeOfDataContentField());
      } else if (conf.data_format() == "channels_last") {
        const int64_t channel_num = dy_blob->shape().At(x_blob->shape().NumAxes() - 1);
        CHECK_EQ(channel_num, bw_buf_blob->shape().At(0));
        KernelUtil<DeviceType::kGPU, T>::RowSum(
            ctx.device_ctx, channel_num, bw_buf_blob->shape().Count(1), bw_buf_blob->dptr<T>(),
            alpha_grad_blob->mut_dptr<T>(), alpha_grad_buf_blob->mut_dptr<T>(),
            alpha_grad_buf_blob->ByteSizeOfDataContentField());
      } else {
        UNIMPLEMENTED();
      }
    }
  }
};

#define INSTANTIATE_P_RELU_ALPHA_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template class PReluAlphaGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_P_RELU_ALPHA_GRAD_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
