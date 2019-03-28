#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

namespace {

inline cudnnBatchNormMode_t CudnnBatchNormModeTraining() {
#if (CUDNN_VERSION >= 7000)
  return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
  return CUDNN_BATCHNORM_SPATIAL;
#endif
}

}  // namespace

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
        ctx->cudnn_handle(), CudnnBatchNormModeTraining(), OnePtr<T>::value, ZeroPtr<T>::value,
        xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(), param_desc.Get(), gamma->dptr<T>(),
        beta->dptr<T>(), 1.0 - momentum, moving_mean->mut_dptr(), moving_variance->mut_dptr(),
        epsilon, mean->mut_dptr(), inv_variance->mut_dptr()));
  }
  static void ForwardInference(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* beta,
                               const Blob* moving_mean, const Blob* moving_variance, Blob* y,
                               Blob* buf, int32_t axis, double epsilon) {
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
    CudnnTensorDesc param_desc(CUDNN_TENSOR_NCHW, data_type, 1, param_dim_size, 1, 1);
    CudaCheck(cudnnBatchNormalizationForwardInference(
        ctx->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, OnePtr<T>::value, ZeroPtr<T>::value,
        xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(), param_desc.Get(), gamma->dptr(),
        beta->dptr(), moving_mean->dptr(), moving_variance->dptr(), epsilon));
  }
  static void Backward(DeviceCtx* ctx, const Blob* x, const Blob* gamma, const Blob* mean,
                       const Blob* inv_variance, const Blob* dy, Blob* dx, Blob* gamma_diff,
                       Blob* beta_diff, Blob* buf, int32_t axis, double epsilon) {
    const int64_t param_dim_size = x->shape().At(axis);
    const DataType data_type = x->data_type();
    CHECK_EQ(dy->shape(), x->shape());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape(), x->shape());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());
    CudnnTensorDesc xy_desc(CUDNN_TENSOR_NCHW, data_type, x->shape().Count(0, axis),
                            x->shape().At(axis), x->shape().Count(axis), 1);
    const auto CheckParamBlob = [&](const Blob* blob) {
      CHECK_EQ(blob->shape().NumAxes(), 1);
      CHECK_EQ(blob->shape().At(0), param_dim_size);
      CHECK_EQ(blob->data_type(), data_type);
    };
    CheckParamBlob(gamma);
    CheckParamBlob(gamma_diff);
    CheckParamBlob(beta_diff);
    CudnnTensorDesc param_desc(CUDNN_TENSOR_NCHW, data_type, 1, param_dim_size, 1, 1);
    CudaCheck(cudnnBatchNormalizationBackward(
        ctx->cudnn_handle(), CudnnBatchNormModeTraining(), OnePtr<T>::value, ZeroPtr<T>::value,
        OnePtr<T>::value, ZeroPtr<T>::value, xy_desc.Get(), x->dptr(), xy_desc.Get(), dy->dptr(),
        xy_desc.Get(), dx->mut_dptr(), param_desc.Get(), gamma->dptr(), gamma_diff->mut_dptr(),
        beta_diff->mut_dptr(), epsilon, mean->dptr(), inv_variance->dptr()));
  }
};

#define INSTANTIATE_NORMALIZATION_KERNEL_UTIL_GPU(type_cpp, type_proto) \
  template struct NormalizationKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_NORMALIZATION_KERNEL_UTIL_GPU, FLOATING_DATA_TYPE_SEQ)
#undef INSTANTIATE_NORMALIZATION_KERNEL_UTIL_GPU

}  // namespace oneflow
