#include "oneflow/core/kernel/prelu_kernel.h"

namespace oneflow {
namespace {
template<typename T>
__global__ void PReluForward(const int64_t elem_cnt, const int64_t channel_num, const int64_t area,
                             const T* in_dptr, const T* alpha_dptr, T* out_dptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    int64_t c = (i / area) % channel_num;
    out_dptr[i] = (in_dptr[i] <= 0) ? in_dptr[i] * alpha_dptr[c] : in_dptr[i];
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
          elem_cnt, 1, 1, in_blob->dptr<T>(), alpha_blob->dptr<T>(), out_blob->mut_dptr<T>());
    } else {
      if (conf.data_format() == "channels_first") {
        const int64_t channel_num = in_blob->shape().At(1);
        const int64_t area = in_blob->shape().Count(2);
        PReluForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                       ctx.device_ctx->cuda_stream()>>>(elem_cnt, channel_num, area,
                                                        in_blob->dptr<T>(), alpha_blob->dptr<T>(),
                                                        out_blob->mut_dptr<T>());
      } else if (conf.data_format() == "channels_last") {
        const int64_t channel_num = in_blob->shape().At(in_blob->shape().NumAxes() - 1);
        PReluForward<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                       ctx.device_ctx->cuda_stream()>>>(elem_cnt, channel_num, 1,
                                                        in_blob->dptr<T>(), alpha_blob->dptr<T>(),
                                                        out_blob->mut_dptr<T>());
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
