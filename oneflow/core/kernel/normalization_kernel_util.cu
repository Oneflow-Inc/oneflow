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
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());
    CudnnTensorDesc xy_desc(CUDNN_TENSOR_NCHW, data_type, x->shape().Count(0, axis),
                            x->shape().At(axis), x->shape().Count(axis), 1);
    const int64_t param_dim_size = x->shape().At(axis);
    const auto CheckParamBlob = [&](const Blob* blob) {
      CHECK_EQ(blob->shape().NumAxes(), 1);
      CHECK_EQ(blob->shape().At(0), param_dim_size);
      CHECK_EQ(blob->data_type(), data_type);
    };
    CheckParamBlob(gamma);
    CheckParamBlob(beta);
    CheckParamBlob(moving_mean);
    CheckParamBlob(moving_variance);
    CheckParamBlob(mean);
    CheckParamBlob(inv_variance);
    CudnnTensorDesc param_desc(CUDNN_TENSOR_NCHW, data_type, 1, param_dim_size, 1, 1);
    CudaCheck(cudnnBatchNormalizationForwardTraining(
        ctx->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, OnePtr<T>::value,
        ZeroPtr<T>::value, xy_desc.Get(), x->dptr<T>(), xy_desc.Get(), y->mut_dptr<T>(),
        param_desc.Get(), gamma->dptr<T>(), beta->dptr<T>(), 1.0 - momentum,
        moving_mean->mut_dptr<T>(), moving_variance->mut_dptr<T>(), epsilon, mean->mut_dptr<T>(),
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
