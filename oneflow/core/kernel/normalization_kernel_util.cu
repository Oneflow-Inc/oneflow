#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

template<typename T>
struct NormalizationKernelUtil<kGPU, T> {
  static void ForwardTraining(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* beta,
                              Blob* y, Blob* moving_mean, Blob* moving_variance, Blob* mean,
                              Blob* inv_variance, Blob* buf, int32_t axis, double epsilon,
                              double momentum) {
    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(x->data_type(), y->data_type());
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());
    CudnnTensorDesc x_desc(CUDNN_TENSOR_NCHW, x->data_type(), x->shape().Count(0, axis), x->shape().At(axis), x->shape().Count(axis), 1);
    CudnnTensorDesc y_desc(CUDNN_TENSOR_NCHW, y->data_type(), x->shape().Count(0, axis), x->shape().At(axis), x->shape().Count(axis), 1);
    
    CudnnTensorDesc(cudnnTensorFormat_t, DataType, int n, int c, int h, int w);

    CudnnTensorDesc in_out_tensor_desc(CUDNN_TENSOR_NCHW, )
    CudaCheck(cudnnBatchNormalizationForwardTraining(
        ctx->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        OnePtr<T>::value, ZeroPtr<T>::value, normalization_ctx_->cudnn_in_tensor_desc(), in,
        normalization_ctx_->cudnn_in_tensor_desc(), out,
        normalization_ctx_->cudnn_param_tensor_desc(), gamma, beta, 1 - momentum, moving_mean,
        moving_variance, epsilon, mean->mut_dptr<T>(),
        inv_variance->mut_dptr<T>()));

  }
  static void ForwardInference(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* beta,
                               const Blob* moving_mean, const Blob* moving_variance, Blob* out,
                               Blob* buf, int32_t axis, double epsilon) {}
  static void Backward(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* mean,
                       const Blob* inv_variance, const Blob* dy, Blob* dx, Blob* gamma_diff,
                       Blob* beta_diff, Blob* buf, int32_t axis, double epsilon) {}
};

}  // namespace oneflow
