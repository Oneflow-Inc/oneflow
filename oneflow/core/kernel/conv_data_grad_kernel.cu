#include "oneflow/core/kernel/conv_data_grad_kernel.h"
#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

template<typename T>
struct ConvDataGradKernelUtil<DeviceType::kGPU, T> final {
  static void Compute(DeviceCtx* ctx, const ConvDataGradKernelConf& kernel_conf,
                      const ConvConf& conf, const Blob* dy, const Blob* filter, Blob* dx, Blob* buf,
                      const bool enable_true_half) {
    CudnnTensorDesc dy_desc(dy->data_type(), dy->shape(), conf.data_format());
    CudnnFilterDesc filter_desc(filter->data_type(), filter->shape(), conf.data_format());
    CudnnTensorDesc dx_desc(dx->data_type(), dx->shape(), conf.data_format());
    CudnnConvDesc conv_desc(GetConvDescDataType(dx->data_type(), enable_true_half), dx->shape(),
                            conf);
    CudaCheck(cudnnConvolutionBackwardData(
        ctx->cudnn_handle(), CudnnSPOnePtr<T>(), filter_desc.Get(), filter->dptr<T>(),
        dy_desc.Get(), dy->dptr<T>(), conv_desc.Get(),
        static_cast<cudnnConvolutionBwdDataAlgo_t>(kernel_conf.cudnn_bwd_data_algo()),
        buf->mut_dptr(), buf->ByteSizeOfDataContentField(), CudnnSPZeroPtr<T>(), dx_desc.Get(),
        dx->mut_dptr<T>()));
  }
};

#define INSTANTIATE_CONV_DATA_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct ConvDataGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_DATA_GRAD_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
