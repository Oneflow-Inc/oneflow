#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

namespace {

#if (CUDNN_VERSION >= 7401)
#define BN_ENABLE_EX_API
#endif

void InferDimSizeAndDataFormat(const ShapeView& x_shape, const int32_t axis, int32_t* n, int32_t* c,
                               int32_t* h, int32_t* w, cudnnTensorFormat_t* format) {
  if (axis != 0 && x_shape.Count(axis + 1) == 1 && CUDNN_VERSION >= 7605) {
    *n = x_shape.At(0);
    *h = x_shape.Count(1, axis);
    *w = 1;
    *c = x_shape.At(axis);
    *format = CUDNN_TENSOR_NHWC;
  } else {
    *n = x_shape.Count(0, axis);
    *c = x_shape.At(axis);
    *h = x_shape.Count(axis + 1);
    *w = 1;
    *format = CUDNN_TENSOR_NCHW;
  }
}

DataType GetParamDataType(const DataType x_data_type) {
  return x_data_type == DataType::kFloat16 ? DataType::kFloat : x_data_type;
}

std::function<void(const user_op::Tensor* tensor)> MakeCheckParamTensorFn(
    const int32_t param_dim_size, const DataType param_data_type) {
  return [=](const user_op::Tensor* tensor) {
    CHECK_EQ(tensor->shape().NumAxes(), 1);
    CHECK_EQ(tensor->shape().At(0), param_dim_size);
    CHECK_EQ(tensor->data_type(), param_data_type);
  };
}

size_t InferTrainWorkspaceSize(const ShapeView& x_shape, const DataType data_type,
                               const int32_t axis) {
#if defined(BN_ENABLE_EX_API)
  int32_t n, c, h, w;
  cudnnTensorFormat_t format;
  InferDimSizeAndDataFormat(x_shape, axis, &n, &c, &h, &w, &format);
  CudnnTensorDesc xy_desc(format, data_type, n, c, h, w);
  CudnnTensorDesc param_desc(format, GetParamDataType(data_type), 1, c, 1, 1);
  size_t size_in_bytes;
  cudnnHandle_t handle;
  CudaCheck(cudnnCreate(&handle));

  CudaCheck(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CUDNN_BATCHNORM_OPS_BN, xy_desc.Get(), nullptr,
      xy_desc.Get(), param_desc.Get(), nullptr, &size_in_bytes));

  CudaCheck(cudnnDestroy(handle));
  return std::max(size_in_bytes, static_cast<size_t>(1));
#else
  return 1;
#endif
}

size_t InferTrainTmpSize(user_op::InferContext* ctx) {
  const auto* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  return InferTrainWorkspaceSize(x->shape(), x->data_type(), axis);
}

size_t InferGradWorkspaceSize(const ShapeView& x_shape, const DataType data_type,
                              const int32_t axis) {
#if defined(BN_ENABLE_EX_API)
  int32_t n, c, h, w;
  cudnnTensorFormat_t format;
  InferDimSizeAndDataFormat(x_shape, axis, &n, &c, &h, &w, &format);
  CudnnTensorDesc xy_desc(format, data_type, n, c, h, w);
  CudnnTensorDesc param_desc(format, GetParamDataType(data_type), 1, c, 1, 1);
  size_t size_in_bytes;
  cudnnHandle_t handle;
  CudaCheck(cudnnCreate(&handle));
  CudaCheck(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CUDNN_BATCHNORM_OPS_BN, xy_desc.Get(), nullptr,
      xy_desc.Get(), nullptr, xy_desc.Get(), param_desc.Get(), nullptr, &size_in_bytes));
  CudaCheck(cudnnDestroy(handle));
  return std::max(size_in_bytes, static_cast<size_t>(1));
#else
  return 1;
#endif
}

size_t InferGradTmpSize(user_op::InferContext* ctx) {
  const auto* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  return InferGradWorkspaceSize(dy->shape(), dy->data_type(), axis);
}

template<typename T>
class NormalizationInferenceKernel final : public user_op::OpKernel {
 public:
  NormalizationInferenceKernel() = default;
  ~NormalizationInferenceKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(!training);
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

    int32_t n, c, h, w;
    cudnnTensorFormat_t format;
    InferDimSizeAndDataFormat(x->shape(), axis, &n, &c, &h, &w, &format);

    CudnnTensorDesc xy_desc(format, data_type, n, c, h, w);
    const DataType param_data_type = GetParamDataType(data_type);
    const auto CheckParamTensor = MakeCheckParamTensorFn(c, param_data_type);
    CheckParamTensor(gamma);
    CheckParamTensor(beta);
    CheckParamTensor(moving_mean);
    CheckParamTensor(moving_variance);
    CudnnTensorDesc param_desc(format, param_data_type, 1, c, 1, 1);

    CudaCheck(cudnnBatchNormalizationForwardInference(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, CudnnSPOnePtr<T>(),
        CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(),
        param_desc.Get(), gamma->dptr(), beta->dptr(), moving_mean->dptr(), moving_variance->dptr(),
        epsilon));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_INFERENCE_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("normalization")                                              \
      .SetCreateFn<NormalizationInferenceKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobAttr<bool>("training") == false));

REGISTER_BN_INFERENCE_KERNEL(float16)
REGISTER_BN_INFERENCE_KERNEL(float)
REGISTER_BN_INFERENCE_KERNEL(double)

#undef REGISTER_BN_INFERENCE_KERNEL

template<typename T>
class NormalizationTrainKernel final : public user_op::OpKernel {
 public:
  NormalizationTrainKernel() = default;
  ~NormalizationTrainKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(training);
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto momentum = ctx->Attr<float>("momentum");
    auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    int32_t n, c, h, w;
    cudnnTensorFormat_t format;
    InferDimSizeAndDataFormat(x->shape(), axis, &n, &c, &h, &w, &format);

    CudnnTensorDesc xy_desc(format, data_type, n, c, h, w);
    const DataType param_data_type = GetParamDataType(data_type);
    const auto CheckParamTensor = MakeCheckParamTensorFn(c, param_data_type);
    CheckParamTensor(gamma);
    CheckParamTensor(beta);
    CheckParamTensor(moving_mean);
    CheckParamTensor(moving_variance);
    CheckParamTensor(mean);
    CheckParamTensor(inv_variance);
    CudnnTensorDesc param_desc(format, param_data_type, 1, c, 1, 1);

#if defined(BN_ENABLE_EX_API)
    size_t workspace_size;
    CudaCheck(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CUDNN_BATCHNORM_OPS_BN, xy_desc.Get(), nullptr, xy_desc.Get(), param_desc.Get(), nullptr,
        &workspace_size));
    size_t reserve_space_size;
    CudaCheck(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CUDNN_BATCHNORM_OPS_BN, nullptr, xy_desc.Get(), &reserve_space_size));
    auto* workspace = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    if (reserve_space_size == 0 && workspace_size <= workspace->shape().elem_cnt()) {
      CudaCheck(cudnnBatchNormalizationForwardTrainingEx(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
          CUDNN_BATCHNORM_OPS_BN, CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(),
          nullptr, nullptr, xy_desc.Get(), y->mut_dptr(), param_desc.Get(), gamma->dptr(),
          beta->dptr(), 1.0 - momentum, moving_mean->mut_dptr(), moving_variance->mut_dptr(),
          epsilon, mean->mut_dptr(), inv_variance->mut_dptr(), nullptr, workspace->mut_dptr(),
          workspace->shape().elem_cnt(), nullptr, 0));
    } else {
      CudaCheck(cudnnBatchNormalizationForwardTraining(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(),
          param_desc.Get(), gamma->dptr(), beta->dptr(), 1.0 - momentum, moving_mean->mut_dptr(),
          moving_variance->mut_dptr(), epsilon, mean->mut_dptr(), inv_variance->mut_dptr()));
    }
#else
    CudaCheck(cudnnBatchNormalizationForwardTraining(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CudnnSPOnePtr<T>(),
        CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(), xy_desc.Get(), y->mut_dptr(),
        param_desc.Get(), gamma->dptr(), beta->dptr(), 1.0 - momentum, moving_mean->mut_dptr(),
        moving_variance->mut_dptr(), epsilon, mean->mut_dptr(), inv_variance->mut_dptr()));
#endif
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class NormalizationGradUserKernel final : public user_op::OpKernel {
 public:
  NormalizationGradUserKernel() = default;
  ~NormalizationGradUserKernel() override = default;

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

    const DataType data_type = x->data_type();
    CHECK_EQ(dy->shape(), x->shape());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape(), x->shape());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    int32_t n, c, h, w;
    cudnnTensorFormat_t format;
    InferDimSizeAndDataFormat(x->shape(), axis, &n, &c, &h, &w, &format);

    CudnnTensorDesc xy_desc(format, data_type, n, c, h, w);
    const DataType param_data_type = GetParamDataType(data_type);
    const auto CheckParamTensor = MakeCheckParamTensorFn(c, param_data_type);
    CheckParamTensor(gamma);
    CheckParamTensor(gamma_diff);
    CheckParamTensor(beta_diff);
    CudnnTensorDesc param_desc(format, param_data_type, 1, c, 1, 1);

#if defined(BN_ENABLE_EX_API)
    size_t workspace_size;
    CudaCheck(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CUDNN_BATCHNORM_OPS_BN, xy_desc.Get(), nullptr, xy_desc.Get(), nullptr, xy_desc.Get(),
        param_desc.Get(), nullptr, &workspace_size));
    size_t reserve_space_size;
    CudaCheck(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CUDNN_BATCHNORM_OPS_BN, nullptr, xy_desc.Get(), &reserve_space_size));
    auto* workspace = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    if (reserve_space_size == 0 && workspace_size <= workspace->shape().elem_cnt()) {
      CudaCheck(cudnnBatchNormalizationBackwardEx(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
          CUDNN_BATCHNORM_OPS_BN, CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(), nullptr, nullptr, xy_desc.Get(),
          dy->dptr(), nullptr, nullptr, xy_desc.Get(), dx->mut_dptr(), param_desc.Get(),
          gamma->dptr(), nullptr, gamma_diff->mut_dptr(), beta_diff->mut_dptr(), epsilon,
          mean->dptr(), inv_variance->dptr(), nullptr, workspace->mut_dptr(),
          workspace->shape().elem_cnt(), nullptr, 0));
    } else {
      CudaCheck(cudnnBatchNormalizationBackward(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(),
          xy_desc.Get(), dy->dptr(), xy_desc.Get(), dx->mut_dptr(), param_desc.Get(), gamma->dptr(),
          gamma_diff->mut_dptr(), beta_diff->mut_dptr(), epsilon, mean->dptr(),
          inv_variance->dptr()));
    }
#else
    CudaCheck(cudnnBatchNormalizationBackward(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CudnnSPOnePtr<T>(),
        CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), xy_desc.Get(), x->dptr(),
        xy_desc.Get(), dy->dptr(), xy_desc.Get(), dx->mut_dptr(), param_desc.Get(), gamma->dptr(),
        gamma_diff->mut_dptr(), beta_diff->mut_dptr(), epsilon, mean->dptr(),
        inv_variance->dptr()));

#endif
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_TRAIN_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("normalization")                                              \
      .SetCreateFn<NormalizationTrainKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobAttr<bool>("training") == true))               \
      .SetInferTmpSizeFn(InferTrainTmpSize);

#define REGISTER_BN_GRAD_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("normalization_grad")                                           \
      .SetCreateFn<NormalizationGradUserKernel<dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kGPU)                  \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferGradTmpSize);

REGISTER_BN_TRAIN_KERNEL(float16)
REGISTER_BN_TRAIN_KERNEL(float)
REGISTER_BN_TRAIN_KERNEL(double)

REGISTER_BN_GRAD_KERNEL(float16)
REGISTER_BN_GRAD_KERNEL(float)
REGISTER_BN_GRAD_KERNEL(double)

}  // namespace
}  // namespace oneflow
