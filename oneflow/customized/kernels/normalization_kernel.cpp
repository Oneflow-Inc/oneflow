#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

namespace {
template<DeviceType device_type, typename T>
class NormalizationUserKernel;

template<typename T>
class NormalizationUserKernel<DeviceType::kCPU, T> final : public user_op::OpKernel {
 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { UNIMPLEMENTED(); };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

inline cudnnBatchNormMode_t CudnnBatchNormModeTraining() {
#if (CUDNN_VERSION >= 7000)
  return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
  return CUDNN_BATCHNORM_SPATIAL;
#endif
}

template<typename T>
class NormalizationUserKernel<DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  NormalizationUserKernel() = default;
  ~NormalizationUserKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());
    CudnnTensorDesc xy_desc(CUDNN_TENSOR_NCHW, data_type, x->shape().Count(0, axis),
                            x->shape().At(axis), x->shape().Count(axis + 1), 1);
    const int64_t param_dim_size = x->shape().At(axis);
    const DataType param_data_type = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
    const auto CheckParamTensor = [&](const user_op::Tensor* tensor) {
      CHECK_EQ(tensor->shape().NumAxes(), 1);
      CHECK_EQ(tensor->shape().At(0), param_dim_size);
      CHECK_EQ(tensor->data_type(), param_data_type);
    };
    CheckParamTensor(gamma);
    CheckParamTensor(beta);
    CheckParamTensor(moving_mean);
    CheckParamTensor(moving_variance);
    CudnnTensorDesc param_desc(CUDNN_TENSOR_NCHW, param_data_type, 1, param_dim_size, 1, 1);

    if (training) {
      const auto momentum = ctx->Attr<float>("momentum");
      auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
      auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
      CheckParamTensor(mean);
      CheckParamTensor(inv_variance);
      CudaCheck(cudnnBatchNormalizationForwardTraining(
          ctx->device_ctx()->cudnn_handle(), CudnnBatchNormModeTraining(), CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(),
          param_desc.Get(), gamma->dptr(), beta->dptr(), 1.0 - momentum, moving_mean->mut_dptr(),
          moving_variance->mut_dptr(), epsilon, mean->mut_dptr(), inv_variance->mut_dptr()));
    } else {
      CudaCheck(cudnnBatchNormalizationForwardInference(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(),
          param_desc.Get(), gamma->dptr(), beta->dptr(), moving_mean->dptr(),
          moving_variance->dptr(), epsilon));
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class NormalizationGradUserKernel;

template<typename T>
class NormalizationGradUserKernel<DeviceType::kCPU, T> final : public user_op::OpKernel {
 private:
  void Compute(user_op::KernelComputeContext* ctx) const override { UNIMPLEMENTED(); }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class NormalizationGradUserKernel<DeviceType::kGPU, T> final : public user_op::OpKernel {
 public:
  NormalizationGradUserKernel() = default;
  ~NormalizationGradUserKernel() = default;

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
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");

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
    const DataType param_data_type = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
    const auto CheckParamTensor = [&](const user_op::Tensor* tensor) {
      CHECK_EQ(tensor->shape().NumAxes(), 1);
      CHECK_EQ(tensor->shape().At(0), param_dim_size);
      CHECK_EQ(tensor->data_type(), param_data_type);
    };
    CheckParamTensor(gamma);
    CheckParamTensor(gamma_diff);
    CheckParamTensor(beta_diff);
    CudnnTensorDesc param_desc(CUDNN_TENSOR_NCHW, param_data_type, 1, param_dim_size, 1, 1);
    CudaCheck(cudnnBatchNormalizationBackward(
        ctx->device_ctx()->cudnn_handle(), CudnnBatchNormModeTraining(), CudnnSPOnePtr<T>(),
        CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(),
        xy_desc.Get(), dy->dptr(), xy_desc.Get(), dx->mut_dptr(), param_desc.Get(), gamma->dptr(),
        gamma_diff->mut_dptr(), beta_diff->mut_dptr(), epsilon, mean->dptr(),
        inv_variance->dptr()));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_KERNEL(op_device_type, dtype)                                   \
  REGISTER_USER_KERNEL("normalization")                                             \
      .SetCreateFn<NormalizationUserKernel<DeviceType::k##op_device_type, dtype>>() \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::k##op_device_type    \
                       & user_op::HobDataType("y", 0) == GetDataType<dtype>::value);

#define REGISTER_BN_GRAD_KERNEL(op_device_type, dtype)                                  \
  REGISTER_USER_KERNEL("normalization_grad")                                            \
      .SetCreateFn<NormalizationGradUserKernel<DeviceType::k##op_device_type, dtype>>() \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::k##op_device_type        \
                       & user_op::HobDataType("dx", 0) == GetDataType<dtype>::value);

REGISTER_BN_KERNEL(CPU, float)
REGISTER_BN_KERNEL(CPU, double)

REGISTER_BN_KERNEL(GPU, float16)
REGISTER_BN_KERNEL(GPU, float)
REGISTER_BN_KERNEL(GPU, double)

REGISTER_BN_GRAD_KERNEL(CPU, float)
REGISTER_BN_GRAD_KERNEL(CPU, double)

REGISTER_BN_GRAD_KERNEL(GPU, float16)
REGISTER_BN_GRAD_KERNEL(GPU, float)
REGISTER_BN_GRAD_KERNEL(GPU, double)

}  // namespace
}  // namespace oneflow
