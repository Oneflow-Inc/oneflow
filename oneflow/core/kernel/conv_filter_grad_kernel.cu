#include "oneflow/core/kernel/conv_filter_grad_kernel.h"
#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

template<typename T>
struct ConvFilterGradKernelUtil<DeviceType::kGPU, T> final {
  static void Compute(DeviceCtx *ctx, const ConvFilterGradKernelConf &kernel_conf,
                      const ConvConf &conf, const Blob *x, const Blob *dy, Blob *filter_diff,
                      Blob *buf, const bool enable_true_half) {
    CudnnTensorDesc x_desc(x->data_type(), x->shape(), conf.data_format());
    CudnnTensorDesc dy_desc(dy->data_type(), dy->shape(), conf.data_format());
    CudnnFilterDesc filter_diff_desc(filter_diff->data_type(), filter_diff->shape(),
                                     conf.data_format());
    CudnnConvDesc conv_desc(GetConvDescDataType(x->data_type(), enable_true_half), x->shape(),
                            conf);
    CudaCheck(cudnnConvolutionBackwardFilter(
        ctx->cudnn_handle(), CudnnSPOnePtr<T>(), x_desc.Get(), x->dptr<T>(), dy_desc.Get(),
        dy->dptr<T>(), conv_desc.Get(),
        static_cast<cudnnConvolutionBwdFilterAlgo_t>(kernel_conf.cudnn_bwd_filter_algo()),
        buf->mut_dptr(), buf->ByteSizeOfDataContentField(), CudnnSPZeroPtr<T>(),
        filter_diff_desc.Get(), filter_diff->mut_dptr<T>()));
  }
};

#define INSTANTIATE_CONV_FILTER_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct ConvFilterGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_FILTER_GRAD_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
