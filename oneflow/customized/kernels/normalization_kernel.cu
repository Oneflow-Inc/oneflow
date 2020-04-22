#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cudnn_util.h"

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
class GpuNormalizationKernel final : public user_op::OpKernel {
 public:
  GpuNormalizationKernel() = default;
  ~GpuNormalizationKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    bool training = true;
    const auto* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    const auto axis = ctx->GetAttr<int32_t>("axis");
    const auto epsilon = ctx->GetAttr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());
    CudnnTensorDesc xy_desc(CUDNN_TENSOR_NCHW, data_type, x->shape().Count(0, axis),
                            x->shape().At(axis), x->shape().Count(axis + 1), 1);
    const int64_t param_dim_size = x->shape().At(axis);
    const auto CheckParamTensor = [&](const user_op::Tensor* tensor) {
      CHECK_EQ(tensor->shape().NumAxes(), 1);
      CHECK_EQ(tensor->shape().At(0), param_dim_size);
      CHECK_EQ(tensor->data_type(), data_type);
    };
    CheckParamTensor(gamma);
    CheckParamTensor(beta);
    CheckParamTensor(moving_mean);
    CheckParamTensor(moving_variance);
    CudnnTensorDesc param_desc(CUDNN_TENSOR_NCHW, data_type, 1, param_dim_size, 1, 1);

    if (training) {
      const auto momentum = ctx->GetAttr<float>("momentum");
      auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
      auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
      CheckParamTensor(mean);
      CheckParamTensor(inv_variance);
      CudaCheck(cudnnBatchNormalizationForwardTraining(
          ctx->device_ctx()->cudnn_handle(), CudnnBatchNormModeTraining(), CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(),
          param_desc.Get(), gamma->dptr<T>(), beta->dptr<T>(), 1.0 - momentum,
          moving_mean->mut_dptr(), moving_variance->mut_dptr(), epsilon, mean->mut_dptr(),
          inv_variance->mut_dptr()));
    } else {
      CudaCheck(cudnnBatchNormalizationForwardInference(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(),
          param_desc.Get(), gamma->dptr<T>(), beta->dptr<T>(), moving_mean->mut_dptr(),
          moving_variance->mut_dptr(), epsilon));
    }
  };
};

#define REGISTER_GPU_KERNEL(dtype)                                                    \
  REGISTER_USER_KERNEL("normalization")                                               \
      .SetCreateFn<GpuNormalizationKernel<dtype>>()                                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && y_desc->data_type() == GetDataType<dtype>::value;                   \
      });

REGISTER_GPU_KERNEL(float)
REGISTER_GPU_KERNEL(double)

template<typename T>
class GpuNormalizationGradKernel final : public user_op::OpKernel {
 public:
  GpuNormalizationGradKernel() = default;
  ~GpuNormalizationGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    auto* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
    auto* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
    const auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const auto axis = ctx->GetAttr<int32_t>("axis");
    const auto epsilon = ctx->GetAttr<float>("epsilon");

    const int64_t param_dim_size = x->shape().At(axis);
    const DataType data_type = x->data_type();
    CHECK_EQ(dy->shape(), x->shape());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape(), x->shape());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());
    CudnnTensorDesc xy_desc(CUDNN_TENSOR_NCHW, data_type, x->shape().Count(0, axis),
                            x->shape().At(axis), x->shape().Count(axis + 1), 1);
    const auto CheckParamTensor = [&](const user_op::Tensor* tensor) {
      CHECK_EQ(tensor->shape().NumAxes(), 1);
      CHECK_EQ(tensor->shape().At(0), param_dim_size);
      CHECK_EQ(tensor->data_type(), data_type);
    };
    CheckParamTensor(gamma);
    CheckParamTensor(gamma_diff);
    CheckParamTensor(beta_diff);
    CudnnTensorDesc param_desc(CUDNN_TENSOR_NCHW, data_type, 1, param_dim_size, 1, 1);
    CudaCheck(cudnnBatchNormalizationBackward(
        ctx->device_ctx()->cudnn_handle(), CudnnBatchNormModeTraining(), CudnnSPOnePtr<T>(),
        CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(),
        xy_desc.Get(), dy->dptr(), xy_desc.Get(), dx->mut_dptr(), param_desc.Get(), gamma->dptr(),
        gamma_diff->mut_dptr(), beta_diff->mut_dptr(), epsilon, mean->dptr(),
        inv_variance->dptr()));
  };
};

#define REGISTER_GPU_GELU_GRAD_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("normalization_grad")                                          \
      .SetCreateFn<GpuNormalizationGradKernel<dtype>>()                               \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_GELU_GRAD_KERNEL(float)
REGISTER_GPU_GELU_GRAD_KERNEL(double)

}  // namespace oneflow
