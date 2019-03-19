#include "oneflow/core/kernel/conv_bias_grad_kernel.h"

namespace oneflow {

template<typename T>
struct ConvBiasGradKernelUtil<DeviceType::kGPU, T> final {
  static void Compute(DeviceCtx* ctx, const std::string& format, const Blob* dy, Blob* bias_diff) {
    CHECK_EQ(bias_diff->shape().NumAxes(), 1);
    CHECK_GE(dy->shape().NumAxes(), 3);
    CHECK_LE(dy->shape().NumAxes(), 5);
    std::unique_ptr<CudnnTensorDesc> dy_desc;
    dy_desc.reset(new CudnnTensorDesc(dy->data_type(), dy->shape(), format));
    std::unique_ptr<CudnnTensorDesc> bias_diff_desc;
    if (format == "channels_first") {
      CHECK_EQ(dy->shape().At(1), bias_diff->shape().At(0));
      dy_desc.reset(new CudnnTensorDesc(CUDNN_TENSOR_NCHW, bias_diff->data_type(), 1,
                                        static_cast<int32_t>(bias_diff->shape().At(0)), 1, 1));
    } else if (format == "channels_last") {
      CHECK_EQ(dy->shape().At(dy->shape().NumAxes() - 1), bias_diff->shape().At(0));
      dy_desc.reset(new CudnnTensorDesc(CUDNN_TENSOR_NHWC, bias_diff->data_type(), 1, 1, 1,
                                        static_cast<int32_t>(bias_diff->shape().At(0))));
    } else {
      UNIMPLEMENTED();
    }
    CudaCheck(cudnnConvolutionBackwardBias(ctx->cudnn_handle(), OnePtr<T>::value, dy_desc->Get(),
                                           dy->dptr<T>(), ZeroPtr<T>::value, bias_diff_desc->Get(),
                                           bias_diff->mut_dptr<T>()));
  }
};

#define INSTANTIATE_CONV_BIAS_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct ConvBiasGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_BIAS_GRAD_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
