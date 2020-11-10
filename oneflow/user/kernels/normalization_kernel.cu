/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifdef WITH_CUDA

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

#if (CUDNN_VERSION >= 7401)
#define BN_ENABLE_EX_API
#endif

void InferDimSizeAndDataFormat(const ShapeView& x_shape, const int32_t axis, int32_t* n, int32_t* c,
                               int32_t* h, int32_t* w, cudnnTensorFormat_t* format) {
  if (x_shape.Count(axis + 1) == 1) {
    if (axis == 0) {
      *n = 1;
      *h = 1;
    } else {
      *n = x_shape.At(0);
      *h = x_shape.Count(1, axis);
    }
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

void InferXYCudnnTensorDesc(const ShapeView& xy_shape, const DataType& data_type,
                            const int32_t axis, cudnnTensorDescriptor_t xy_desc) {
  int32_t n, c, h, w;
  cudnnTensorFormat_t format;
  InferDimSizeAndDataFormat(xy_shape, axis, &n, &c, &h, &w, &format);
  OF_CUDNN_CHECK(
      cudnnSetTensor4dDescriptor(xy_desc, format, GetCudnnDataType(data_type), n, c, h, w));
}

void InferParamCudnnTensorDesc(const cudnnTensorDescriptor_t xy_desc, cudnnBatchNormMode_t mode,
                               cudnnTensorDescriptor_t param_desc) {
  OF_CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(param_desc, xy_desc, mode));
}

class CudnnTensorDescHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnTensorDescHelper);
  CudnnTensorDescHelper(const ShapeView& xy_shape, const DataType& data_type, const int32_t axis,
                        cudnnBatchNormMode_t mode) {
    OF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&xy_desc_));
    InferXYCudnnTensorDesc(xy_shape, data_type, axis, xy_desc_);
    OF_CUDNN_CHECK(cudnnCreateTensorDescriptor(&param_desc_));
    InferParamCudnnTensorDesc(xy_desc_, mode, param_desc_);
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    OF_CUDNN_CHECK(cudnnGetTensor4dDescriptor(param_desc_, &param_data_type_, &n, &c, &h, &w,
                                              &n_stride, &c_stride, &h_stride, &w_stride));
    param_size_ = c;
  }
  ~CudnnTensorDescHelper() {
    OF_CUDNN_CHECK(cudnnDestroyTensorDescriptor(param_desc_));
    OF_CUDNN_CHECK(cudnnDestroyTensorDescriptor(xy_desc_));
  }

  cudnnTensorDescriptor_t xy_desc() const { return xy_desc_; }

  cudnnTensorDescriptor_t param_desc() const { return param_desc_; }

  void CheckParamTensor(const user_op::Tensor* tensor) const {
    CHECK_EQ(tensor->shape().NumAxes(), 1);
    CHECK_EQ(tensor->shape().At(0), param_size_);
    CHECK_EQ(GetCudnnDataType(tensor->data_type()), param_data_type_);
  }

 private:
  cudnnTensorDescriptor_t xy_desc_ = nullptr;
  cudnnTensorDescriptor_t param_desc_ = nullptr;
  cudnnDataType_t param_data_type_;
  int32_t param_size_ = 0;
};

size_t InferTrainWorkspaceSize(const ShapeView& x_shape, const DataType data_type,
                               const int32_t axis) {
#if defined(BN_ENABLE_EX_API)
  const CudnnTensorDescHelper desc_helper(x_shape, data_type, axis,
                                          CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  size_t size_in_bytes;
  cudnnHandle_t handle;
  OF_CUDNN_CHECK(cudnnCreate(&handle));
  OF_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CUDNN_BATCHNORM_OPS_BN, desc_helper.xy_desc(),
      nullptr, desc_helper.xy_desc(), desc_helper.param_desc(), nullptr, &size_in_bytes));
  OF_CUDNN_CHECK(cudnnDestroy(handle));
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
  const CudnnTensorDescHelper desc_helper(x_shape, data_type, axis,
                                          CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  size_t size_in_bytes;
  cudnnHandle_t handle;
  OF_CUDNN_CHECK(cudnnCreate(&handle));
  OF_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CUDNN_BATCHNORM_OPS_BN, desc_helper.xy_desc(),
      nullptr, desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(), desc_helper.param_desc(),
      nullptr, &size_in_bytes));
  OF_CUDNN_CHECK(cudnnDestroy(handle));
  return std::max(size_in_bytes, static_cast<size_t>(1));
#else
  return 1;
#endif
}

size_t InferGradTmpSize(user_op::InferContext* ctx) {
  const auto* dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  size_t tmp_size = 0;
  if (ctx->user_op_conf().op_type_name() == "normalization_add_relu_grad"
      && !ctx->user_op_conf().has_output("addend_diff", 0)) {
    tmp_size += GetCudaAlignedSize(dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
  }
  tmp_size += GetCudaAlignedSize(InferGradWorkspaceSize(dy->shape(), dy->data_type(), axis));
  return tmp_size;
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

    const CudnnTensorDescHelper desc_helper(x->shape(), data_type, axis, CUDNN_BATCHNORM_SPATIAL);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(beta);
    desc_helper.CheckParamTensor(moving_mean);
    desc_helper.CheckParamTensor(moving_variance);

    const void* sp_alpha = CudnnSPOnePtr<T>();
    const void* sp_beta;
    if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), y->data_type());
      CHECK_EQ(add_to_output->shape(), y->shape());
      Memcpy<DeviceType::kGPU>(
          ctx->device_ctx(), y->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      sp_beta = CudnnSPOnePtr<T>();
    } else {
      sp_beta = CudnnSPZeroPtr<T>();
    }

    OF_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, sp_alpha, sp_beta,
        desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
        desc_helper.param_desc(), gamma->dptr(), beta->dptr(), moving_mean->dptr(),
        moving_variance->dptr(), epsilon));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_INFERENCE_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationInferenceKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)            \
                       & (user_op::HobAttr<bool>("training") == false))                         \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.user_op_conf().has_input("_add_to_output", 0)) {                                \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));           \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_BN_INFERENCE_KERNEL(float16)
REGISTER_BN_INFERENCE_KERNEL(float)
REGISTER_BN_INFERENCE_KERNEL(double)

#undef REGISTER_BN_INFERENCE_KERNEL

constexpr int64_t kCudaWarpSize = 32;

template<typename T>
__global__ void ReluGpu(int64_t n, const T* x, T* y, int32_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T x_val = x[i];
    const bool is_positive = (x_val > 0);
    int32_t warp_mask = __ballot_sync(__activemask(), static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? x_val : 0;
  }
}

template<>
__global__ void ReluGpu<half>(int64_t n, const half* x, half* y, int32_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  const half zero = __float2half(0.0f);
  CUDA_1D_KERNEL_LOOP(i, n) {
    const half x_val = x[i];
    const bool is_positive = __hgt(x_val, zero);
    int32_t warp_mask = __ballot_sync(__activemask(), static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? x_val : zero;
  }
}

template<typename T>
__global__ void AddReluGpu(int64_t n, const T* x, const T* addend, T* y, int32_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T sum = x[i] + addend[i];
    const bool is_positive = (sum > 0);
    int32_t warp_mask = __ballot_sync(__activemask(), static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? sum : 0;
  }
}

template<>
__global__ void AddReluGpu<half>(int64_t n, const half* x, const half* addend, half* y,
                                 int32_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  const half zero = __float2half(0.0f);
  CUDA_1D_KERNEL_LOOP(i, n) {
    const half sum = __hadd(x[i], addend[i]);
    const bool is_positive = __hgt(sum, zero);
    int32_t warp_mask = __ballot_sync(__activemask(), static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? sum : zero;
  }
}

template<typename T>
void Relu(DeviceCtx* device_ctx, int64_t n, const T* x, T* y, int32_t* mask) {
  ReluGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, device_ctx->cuda_stream()>>>(
      n, x, y, mask);
}

template<>
void Relu<float16>(DeviceCtx* device_ctx, int64_t n, const float16* x, float16* y, int32_t* mask) {
  Relu<half>(device_ctx, n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y), mask);
}

template<typename T>
void AddRelu(DeviceCtx* device_ctx, int64_t n, const T* x, const T* addend, T* y, int32_t* mask) {
  AddReluGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, device_ctx->cuda_stream()>>>(
      n, x, addend, y, mask);
}

template<>
void AddRelu<float16>(DeviceCtx* device_ctx, int64_t n, const float16* x, const float16* addend,
                      float16* y, int32_t* mask) {
  AddRelu<half>(device_ctx, n, reinterpret_cast<const half*>(x),
                reinterpret_cast<const half*>(addend), reinterpret_cast<half*>(y), mask);
}

template<typename T>
__global__ void ReluBackwardGpu(int64_t n, const int32_t* mask, const T* dy, T* addend_diff) {
  int32_t lane_id = threadIdx.x % kCudaWarpSize;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int32_t mask_val = mask[i / kCudaWarpSize];
    bool is_positive = mask_val & (1 << lane_id);
    addend_diff[i] = static_cast<T>(is_positive) * dy[i];
  }
}

template<typename T>
void ReluBackward(DeviceCtx* device_ctx, int64_t n, const int32_t* mask, const T* dy,
                  T* addend_diff) {
  ReluBackwardGpu<T>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, device_ctx->cuda_stream()>>>(
          n, mask, dy, addend_diff);
}

template<>
void ReluBackward<float16>(DeviceCtx* device_ctx, int64_t n, const int32_t* mask, const float16* dy,
                           float16* addend_diff) {
  ReluBackward<half>(device_ctx, n, mask, reinterpret_cast<const half*>(dy),
                     reinterpret_cast<half*>(addend_diff));
}

template<typename T>
class NormalizationTrainKernel final : public user_op::OpKernel {
 public:
  NormalizationTrainKernel() = default;
  ~NormalizationTrainKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    if (ctx->user_op_conf().op_type_name() == "normalization") {
      CHECK(ctx->Attr<bool>("training"));
    }
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

    const CudnnTensorDescHelper desc_helper(x->shape(), data_type, axis,
                                            CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(beta);
    desc_helper.CheckParamTensor(moving_mean);
    desc_helper.CheckParamTensor(moving_variance);
    desc_helper.CheckParamTensor(mean);
    desc_helper.CheckParamTensor(inv_variance);

    const void* sp_alpha = CudnnSPOnePtr<T>();
    const void* sp_beta;
    if (ctx->user_op_conf().has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), y->data_type());
      CHECK_EQ(add_to_output->shape(), y->shape());
      Memcpy<DeviceType::kGPU>(
          ctx->device_ctx(), y->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      sp_beta = CudnnSPOnePtr<T>();
    } else {
      sp_beta = CudnnSPZeroPtr<T>();
    }

#if defined(BN_ENABLE_EX_API)
    size_t workspace_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CUDNN_BATCHNORM_OPS_BN, desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(),
        desc_helper.param_desc(), nullptr, &workspace_size));
    size_t reserve_space_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CUDNN_BATCHNORM_OPS_BN, nullptr, desc_helper.xy_desc(), &reserve_space_size));
    auto* workspace = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    if (reserve_space_size == 0 && workspace_size <= workspace->shape().elem_cnt()) {
      OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
          CUDNN_BATCHNORM_OPS_BN, sp_alpha, sp_beta, desc_helper.xy_desc(), x->dptr(), nullptr,
          nullptr, desc_helper.xy_desc(), y->mut_dptr(), desc_helper.param_desc(), gamma->dptr(),
          beta->dptr(), 1.0 - momentum, moving_mean->mut_dptr(), moving_variance->mut_dptr(),
          epsilon, mean->mut_dptr(), inv_variance->mut_dptr(), nullptr, workspace->mut_dptr(),
          workspace->shape().elem_cnt(), nullptr, 0));
    } else {
      OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, sp_alpha, sp_beta,
          desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
          desc_helper.param_desc(), gamma->dptr(), beta->dptr(), 1.0 - momentum,
          moving_mean->mut_dptr(), moving_variance->mut_dptr(), epsilon, mean->mut_dptr(),
          inv_variance->mut_dptr()));
    }
#else
    OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, sp_alpha, sp_beta,
        desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
        desc_helper.param_desc(), gamma->dptr(), beta->dptr(), 1.0 - momentum,
        moving_mean->mut_dptr(), moving_variance->mut_dptr(), epsilon, mean->mut_dptr(),
        inv_variance->mut_dptr()));
#endif

    if (ctx->user_op_conf().op_type_name() == "normalization_add_relu") {
      CHECK(!ctx->user_op_conf().has_input("_add_to_output", 0));
      const int64_t elem_cnt = x->shape().elem_cnt();
      auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
      if (ctx->user_op_conf().has_input("addend", 0)) {
        const auto* addend = ctx->Tensor4ArgNameAndIndex("addend", 0);
        AddRelu(ctx->device_ctx(), elem_cnt, y->dptr<T>(), addend->dptr<T>(), y->mut_dptr<T>(),
                mask->mut_dptr<int32_t>());
      } else {
        Relu(ctx->device_ctx(), elem_cnt, y->dptr<T>(), y->mut_dptr<T>(),
             mask->mut_dptr<int32_t>());
      }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_TRAIN_KERNEL(dtype)                                                         \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationTrainKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)            \
                       & (user_op::HobAttr<bool>("training") == true))                          \
      .SetInferTmpSizeFn(InferTrainTmpSize)                                                     \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.user_op_conf().has_input("_add_to_output", 0)) {                                \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));           \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_BN_TRAIN_KERNEL(float16)
REGISTER_BN_TRAIN_KERNEL(float)
REGISTER_BN_TRAIN_KERNEL(double)

#define REGISTER_BN_ADD_RELU_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("normalization_add_relu")                                      \
      .SetCreateFn<NormalizationTrainKernel<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                             \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTrainTmpSize);

REGISTER_BN_ADD_RELU_KERNEL(float16)
REGISTER_BN_ADD_RELU_KERNEL(float)
REGISTER_BN_ADD_RELU_KERNEL(double)

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
    auto* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(dy->shape(), x->shape());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape(), x->shape());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    const CudnnTensorDescHelper desc_helper(x->shape(), data_type, axis,
                                            CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(gamma_diff);
    desc_helper.CheckParamTensor(beta_diff);
    desc_helper.CheckParamTensor(mean);
    desc_helper.CheckParamTensor(inv_variance);

    void* bn_workspace_ptr;
    size_t bn_workspace_size;
    const void* bn_dy_ptr;

    if (ctx->user_op_conf().op_type_name() == "normalization_grad") {
      bn_workspace_ptr = tmp_buffer->mut_dptr();
      bn_workspace_size = tmp_buffer->shape().elem_cnt();
      bn_dy_ptr = dy->dptr();
    } else if (ctx->user_op_conf().op_type_name() == "normalization_add_relu_grad") {
      const int64_t elem_cnt = dy->shape().elem_cnt();
      const auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
      user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
      if (ctx->user_op_conf().has_output("addend_diff", 0)) {
        user_op::Tensor* addend_diff = ctx->Tensor4ArgNameAndIndex("addend_diff", 0);
        ReluBackward(ctx->device_ctx(), elem_cnt, mask->dptr<int32_t>(), dy->dptr<T>(),
                     addend_diff->mut_dptr<T>());
        bn_workspace_ptr = tmp_buffer->mut_dptr();
        bn_workspace_size = tmp_buffer->shape().elem_cnt();
        bn_dy_ptr = addend_diff->dptr();
      } else {
        const size_t tmp_buffer_size = tmp_buffer->shape().elem_cnt();
        const size_t relu_dx_size =
            GetCudaAlignedSize(dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
        CHECK_GE(tmp_buffer_size, relu_dx_size);
        ReluBackward(ctx->device_ctx(), elem_cnt, mask->dptr<int32_t>(), dy->dptr<T>(),
                     reinterpret_cast<T*>(tmp_buffer->mut_dptr()));
        bn_workspace_ptr = tmp_buffer->mut_dptr<char>() + relu_dx_size;
        bn_workspace_size = tmp_buffer_size - relu_dx_size;
        bn_dy_ptr = tmp_buffer->dptr();
      }
    } else {
      UNIMPLEMENTED();
    }

#if defined(BN_ENABLE_EX_API)
    size_t workspace_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CUDNN_BATCHNORM_OPS_BN, desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(), nullptr,
        desc_helper.xy_desc(), desc_helper.param_desc(), nullptr, &workspace_size));
    size_t reserve_space_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CUDNN_BATCHNORM_OPS_BN, nullptr, desc_helper.xy_desc(), &reserve_space_size));
    if (reserve_space_size == 0 && workspace_size <= bn_workspace_size) {
      OF_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
          CUDNN_BATCHNORM_OPS_BN, CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), desc_helper.xy_desc(), x->dptr(), nullptr, nullptr,
          desc_helper.xy_desc(), bn_dy_ptr, nullptr, nullptr, desc_helper.xy_desc(), dx->mut_dptr(),
          desc_helper.param_desc(), gamma->dptr(), nullptr, gamma_diff->mut_dptr(),
          beta_diff->mut_dptr(), epsilon, mean->dptr(), inv_variance->dptr(), nullptr,
          bn_workspace_ptr, bn_workspace_size, nullptr, 0));
    } else {
      OF_CUDNN_CHECK(cudnnBatchNormalizationBackward(
          ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CudnnSPOnePtr<T>(),
          CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), desc_helper.xy_desc(),
          x->dptr(), desc_helper.xy_desc(), bn_dy_ptr, desc_helper.xy_desc(), dx->mut_dptr(),
          desc_helper.param_desc(), gamma->dptr(), gamma_diff->mut_dptr(), beta_diff->mut_dptr(),
          epsilon, mean->dptr(), inv_variance->dptr()));
    }
#else
    OF_CUDNN_CHECK(cudnnBatchNormalizationBackward(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, CudnnSPOnePtr<T>(),
        CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), desc_helper.xy_desc(),
        x->dptr(), desc_helper.xy_desc(), bn_dy_ptr, desc_helper.xy_desc(), dx->mut_dptr(),
        desc_helper.param_desc(), gamma->dptr(), gamma_diff->mut_dptr(), beta_diff->mut_dptr(),
        epsilon, mean->dptr(), inv_variance->dptr()));
#endif
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_GRAD_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("normalization_grad")                                           \
      .SetCreateFn<NormalizationGradUserKernel<dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferGradTmpSize);

REGISTER_BN_GRAD_KERNEL(float16)
REGISTER_BN_GRAD_KERNEL(float)
REGISTER_BN_GRAD_KERNEL(double)

#define REGISTER_BN_ADD_RELU_GRAD_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("normalization_add_relu_grad")                                  \
      .SetCreateFn<NormalizationGradUserKernel<dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferGradTmpSize);

REGISTER_BN_ADD_RELU_GRAD_KERNEL(float16)
REGISTER_BN_ADD_RELU_GRAD_KERNEL(float)
REGISTER_BN_ADD_RELU_GRAD_KERNEL(double)

#if (CUDNN_VERSION >= 7401)

size_t InferFusedNormalizationAddReluTmpSize(user_op::InferContext* ctx) {
  const auto* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  const CudnnTensorDescHelper desc_helper(x->shape(), x->data_type(), axis,
                                          CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  size_t size_in_bytes;
  cudnnHandle_t handle;
  OF_CUDNN_CHECK(cudnnCreate(&handle));
  CudnnActivationDesc activation_desc(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);
  cudnnBatchNormOps_t ops;
  cudnnTensorDescriptor_t z_desc;
  if (ctx->user_op_conf().has_input("addend", 0)) {
    ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    z_desc = desc_helper.xy_desc();
  } else {
    ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    z_desc = nullptr;
  }
  OF_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops, desc_helper.xy_desc(), z_desc,
      desc_helper.xy_desc(), desc_helper.param_desc(), activation_desc.Get(), &size_in_bytes));
  OF_CUDNN_CHECK(cudnnDestroy(handle));
  return std::max(size_in_bytes, static_cast<size_t>(1));
}

size_t InferFusedNormalizationAddReluGradTmpSize(user_op::InferContext* ctx) {
  const auto* x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  const CudnnTensorDescHelper desc_helper(x->shape(), x->data_type(), axis,
                                          CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  size_t size_in_bytes;
  cudnnHandle_t handle;
  OF_CUDNN_CHECK(cudnnCreate(&handle));
  CudnnActivationDesc activation_desc(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);
  cudnnBatchNormOps_t ops;
  cudnnTensorDescriptor_t z_desc;
  if (ctx->user_op_conf().has_output("addend_diff", 0)) {
    ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    z_desc = desc_helper.xy_desc();
  } else {
    ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    z_desc = nullptr;
  }
  OF_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops, desc_helper.xy_desc(), desc_helper.xy_desc(),
      desc_helper.xy_desc(), z_desc, desc_helper.xy_desc(), desc_helper.param_desc(),
      activation_desc.Get(), &size_in_bytes));
  OF_CUDNN_CHECK(cudnnDestroy(handle));
  return std::max(size_in_bytes, static_cast<size_t>(1));
}

template<typename T>
class FusedNormalizationAddReluKernel final : public user_op::OpKernel {
 public:
  FusedNormalizationAddReluKernel() = default;
  ~FusedNormalizationAddReluKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
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
    auto* reserve_space = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
    auto* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    const CudnnTensorDescHelper desc_helper(x->shape(), data_type, axis,
                                            CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(beta);
    desc_helper.CheckParamTensor(moving_mean);
    desc_helper.CheckParamTensor(moving_variance);
    desc_helper.CheckParamTensor(mean);
    desc_helper.CheckParamTensor(inv_variance);

    CudnnActivationDesc activation_desc(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);
    cudnnTensorDescriptor_t z_desc;
    const void* z_ptr;
    cudnnBatchNormOps_t ops;
    if (ctx->user_op_conf().has_input("addend", 0)) {
      z_desc = desc_helper.xy_desc();
      z_ptr = ctx->Tensor4ArgNameAndIndex("addend", 0)->dptr();
      ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
      z_desc = nullptr;
      z_ptr = nullptr;
      ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    }

    size_t min_workspace_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops,
        desc_helper.xy_desc(), z_desc, desc_helper.xy_desc(), desc_helper.param_desc(),
        activation_desc.Get(), &min_workspace_size));
    const size_t workspace_size = tmp_buffer->shape().elem_cnt();
    CHECK_GE(workspace_size, min_workspace_size);
    size_t min_reserve_space_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops,
        activation_desc.Get(), desc_helper.xy_desc(), &min_reserve_space_size));
    const size_t reserve_space_size = reserve_space->shape().elem_cnt();
    CHECK_GE(reserve_space_size, min_reserve_space_size);

    OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops,
        CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), desc_helper.xy_desc(), x->dptr(), z_desc, z_ptr,
        desc_helper.xy_desc(), y->mut_dptr(), desc_helper.param_desc(), gamma->dptr(), beta->dptr(),
        1.0 - momentum, moving_mean->mut_dptr(), moving_variance->mut_dptr(), epsilon,
        mean->mut_dptr(), inv_variance->mut_dptr(), activation_desc.Get(), tmp_buffer->mut_dptr(),
        workspace_size, reserve_space->mut_dptr(), reserve_space_size));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BN_ADD_RELU_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("cudnn_fused_normalization_add_relu")                          \
      .SetCreateFn<FusedNormalizationAddReluKernel<dtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                             \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferFusedNormalizationAddReluTmpSize);

REGISTER_FUSED_BN_ADD_RELU_KERNEL(float16)

template<typename T>
class FusedNormalizationAddReluGradUserKernel final : public user_op::OpKernel {
 public:
  FusedNormalizationAddReluGradUserKernel() = default;
  ~FusedNormalizationAddReluGradUserKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
    auto* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
    const auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const auto* reserve_space = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
    auto* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(dy->shape(), x->shape());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape(), x->shape());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    const CudnnTensorDescHelper desc_helper(x->shape(), data_type, axis,
                                            CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(beta);
    desc_helper.CheckParamTensor(gamma_diff);
    desc_helper.CheckParamTensor(beta_diff);
    desc_helper.CheckParamTensor(mean);
    desc_helper.CheckParamTensor(inv_variance);

    CudnnActivationDesc activation_desc(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);
    cudnnTensorDescriptor_t dz_desc;
    void* dz_ptr;
    cudnnBatchNormOps_t ops;
    if (ctx->user_op_conf().has_output("addend_diff", 0)) {
      dz_desc = desc_helper.xy_desc();
      dz_ptr = ctx->Tensor4ArgNameAndIndex("addend_diff", 0)->mut_dptr();
      ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    } else {
      dz_desc = nullptr;
      dz_ptr = nullptr;
      ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    }

    size_t min_workspace_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops,
        desc_helper.xy_desc(), desc_helper.xy_desc(), desc_helper.xy_desc(), dz_desc,
        desc_helper.xy_desc(), desc_helper.param_desc(), activation_desc.Get(),
        &min_workspace_size));
    const size_t workspace_size = tmp_buffer->shape().elem_cnt();
    CHECK_GE(workspace_size, min_workspace_size);
    size_t min_reserve_space_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops,
        activation_desc.Get(), desc_helper.xy_desc(), &min_reserve_space_size));
    const size_t reserve_space_size = reserve_space->shape().elem_cnt();
    CHECK_GE(reserve_space_size, min_reserve_space_size);
    OF_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
        ctx->device_ctx()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops,
        CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(),
        desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->dptr(), desc_helper.xy_desc(),
        dy->dptr(), dz_desc, dz_ptr, desc_helper.xy_desc(), dx->mut_dptr(),
        desc_helper.param_desc(), gamma->dptr(), beta->dptr(), gamma_diff->mut_dptr(),
        beta_diff->mut_dptr(), epsilon, mean->dptr(), inv_variance->dptr(), activation_desc.Get(),
        tmp_buffer->mut_dptr(), workspace_size, const_cast<void*>(reserve_space->dptr()),
        reserve_space_size));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_BN_ADD_RELU_GRAD_KERNEL(dtype)                                  \
  REGISTER_USER_KERNEL("cudnn_fused_normalization_add_relu_grad")                      \
      .SetCreateFn<FusedNormalizationAddReluGradUserKernel<dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                              \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferFusedNormalizationAddReluGradTmpSize);

REGISTER_FUSED_BN_ADD_RELU_GRAD_KERNEL(float16)

#endif

}  // namespace
}  // namespace oneflow

#endif
