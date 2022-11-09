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

#include <unordered_map>

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#include <cudnn.h>

#if (CUDNN_VERSION >= 7401)
#define BN_ENABLE_EX_API
#endif

namespace oneflow {

namespace {

cudnnBatchNormMode_t getCudnnBatchNormMode(const int64_t dim) {
  if (dim == 2) {
    return CUDNN_BATCHNORM_PER_ACTIVATION;
  } else if (ParseBooleanFromEnv("ONEFLOW_ENABLE_NHWC", false)) {
    return CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  } else {
    // NOTE(Liang Depeng): The new CUDNN_BATCHNORM_SPATIAL_PERSISTENT mode was
    // introduced in CuDNN 7 for performance optimization, but it results in
    // accuracy losses in convolution models such as ResNeXt-101 and
    // video R(2+1)D. We will fall back to the normal CUDNN_BATCHNORM_SPATIAL
    return CUDNN_BATCHNORM_SPATIAL;
  }
}

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
    CHECK_NOTNULL(tensor);
    CHECK_EQ(tensor->shape_view().NumAxes(), 1);
    CHECK_EQ(tensor->shape_view().At(0), param_size_);
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
  cudnnBatchNormMode_t mode = getCudnnBatchNormMode(x_shape.NumAxes());
  const CudnnTensorDescHelper desc_helper(x_shape, data_type, axis, mode);
  size_t size_in_bytes;
  cudnnHandle_t handle = Singleton<CudnnHandlePool>::Get()->Get();
  OF_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle, mode, CUDNN_BATCHNORM_OPS_BN, desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(),
      desc_helper.param_desc(), nullptr, &size_in_bytes));
  Singleton<CudnnHandlePool>::Get()->Put(handle);
  return std::max(size_in_bytes, static_cast<size_t>(1));
#else
  return 1;
#endif
}

size_t InferTrainTmpSize(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("x", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  return InferTrainWorkspaceSize(x.shape(), x.data_type(), axis);
}

size_t InferGradWorkspaceSize(const ShapeView& x_shape, const DataType data_type,
                              const int32_t axis) {
#if defined(BN_ENABLE_EX_API)
  cudnnBatchNormMode_t mode = getCudnnBatchNormMode(x_shape.NumAxes());
  const CudnnTensorDescHelper desc_helper(x_shape, data_type, axis, mode);
  size_t size_in_bytes;
  cudnnHandle_t handle = Singleton<CudnnHandlePool>::Get()->Get();
  OF_CUDNN_CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(
      handle, mode, CUDNN_BATCHNORM_OPS_BN, desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(),
      nullptr, desc_helper.xy_desc(), desc_helper.param_desc(), nullptr, &size_in_bytes));
  Singleton<CudnnHandlePool>::Get()->Put(handle);
  return std::max(size_in_bytes, static_cast<size_t>(1));
#else
  return 1;
#endif
}

size_t InferGradTmpSize(user_op::InferContext* ctx) {
  const auto& dy = ctx->InputTensorDesc("dy", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  size_t tmp_size = 0;
  if (ctx->op_type_name() == "normalization_add_relu_grad" && !ctx->has_output("addend_diff", 0)) {
    tmp_size += GetCudaAlignedSize(dy.shape().elem_cnt() * GetSizeOfDataType(dy.data_type()));
  }
  tmp_size += GetCudaAlignedSize(InferGradWorkspaceSize(dy.shape(), dy.data_type(), axis));
  return tmp_size;
}

class NormalizationInferenceKernel final : public user_op::OpKernel,
                                           public user_op::CudaGraphSupport {
 public:
  NormalizationInferenceKernel() = default;
  ~NormalizationInferenceKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
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
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

    cudnnBatchNormMode_t mode = getCudnnBatchNormMode(x->shape_view().NumAxes());
    const CudnnTensorDescHelper desc_helper(x->shape_view(), data_type, axis, mode);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(beta);
    desc_helper.CheckParamTensor(moving_mean);
    desc_helper.CheckParamTensor(moving_variance);

    const void* sp_alpha = CudnnSPOnePtr(data_type);
    const void* sp_beta;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), y->data_type());
      CHECK_EQ(add_to_output->shape_view(), y->shape_view());
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), y->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      sp_beta = CudnnSPOnePtr(data_type);
    } else {
      sp_beta = CudnnSPZeroPtr(data_type);
    }

    OF_CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, sp_alpha, sp_beta,
        desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
        desc_helper.param_desc(), gamma->dptr(), beta->dptr(), moving_mean->dptr(),
        moving_variance->dptr(), epsilon));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("normalization")
    .SetCreateFn<NormalizationInferenceKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobAttr<bool>("training") == false))
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    });

constexpr int64_t kCudaWarpSize = 32;

template<typename T>
__global__ void ReluGpu(int64_t n, const T* x, T* y, int32_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  const T zero = static_cast<T>(0.f);
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T x_val = x[i];
    const bool is_positive = (x_val > zero);
    int32_t warp_mask = __ballot_sync(__activemask(), static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? x_val : zero;
  }
}

template<typename T>
__global__ void AddReluGpu(int64_t n, const T* x, const T* addend, T* y, int32_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  const T zero = static_cast<T>(0.f);
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T sum = x[i] + addend[i];
    const bool is_positive = (sum > zero);
    int32_t warp_mask = __ballot_sync(__activemask(), static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? sum : zero;
  }
}

template<typename T>
void Relu(ep::Stream* stream, int64_t n, const T* x, T* y, int32_t* mask) {
  ReluGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
               stream->As<ep::CudaStream>()->cuda_stream()>>>(n, x, y, mask);
}

template<typename T>
void AddRelu(ep::Stream* stream, int64_t n, const T* x, const T* addend, T* y, int32_t* mask) {
  AddReluGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                  stream->As<ep::CudaStream>()->cuda_stream()>>>(n, x, addend, y, mask);
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

#if CUDA_VERSION >= 11000

template<>
__global__ void ReluBackwardGpu<nv_bfloat16>(int64_t n, const int32_t* mask, const nv_bfloat16* dy,
                                             nv_bfloat16* addend_diff) {
  int32_t lane_id = threadIdx.x % kCudaWarpSize;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int32_t mask_val = mask[i / kCudaWarpSize];
    bool is_positive = mask_val & (1 << lane_id);
    addend_diff[i] = static_cast<nv_bfloat16>(static_cast<float>(is_positive)) * dy[i];
  }
}

#endif

template<typename T>
void ReluBackward(ep::Stream* stream, int64_t n, const int32_t* mask, const T* dy, T* addend_diff) {
  ReluBackwardGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                       stream->As<ep::CudaStream>()->cuda_stream()>>>(n, mask, dy, addend_diff);
}

void Relu(ep::Stream* stream, int64_t n, const DataType data_type, const void* x, void* y,
          int32_t* mask) {
  if (data_type == kFloat) {
    Relu<float>(stream, n, reinterpret_cast<const float*>(x), reinterpret_cast<float*>(y), mask);
  } else if (data_type == kDouble) {
    Relu<double>(stream, n, reinterpret_cast<const double*>(x), reinterpret_cast<double*>(y), mask);
  } else if (data_type == kFloat16) {
    Relu<half>(stream, n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y), mask);
  } else if (data_type == kBFloat16) {
#if CUDA_VERSION >= 11000
    Relu<nv_bfloat16>(stream, n, reinterpret_cast<const nv_bfloat16*>(x),
                      reinterpret_cast<nv_bfloat16*>(y), mask);
#else
    UNIMPLEMENTED();
#endif
  } else {
    UNIMPLEMENTED();
  }
}
void AddRelu(ep::Stream* stream, int64_t n, const DataType data_type, const void* x,
             const void* addend, void* y, int32_t* mask) {
  if (data_type == kFloat) {
    AddRelu<float>(stream, n, reinterpret_cast<const float*>(x),
                   reinterpret_cast<const float*>(addend), reinterpret_cast<float*>(y), mask);
  } else if (data_type == kDouble) {
    AddRelu<double>(stream, n, reinterpret_cast<const double*>(x),
                    reinterpret_cast<const double*>(addend), reinterpret_cast<double*>(y), mask);
  } else if (data_type == kFloat16) {
    AddRelu<half>(stream, n, reinterpret_cast<const half*>(x),
                  reinterpret_cast<const half*>(addend), reinterpret_cast<half*>(y), mask);
  } else if (data_type == kBFloat16) {
#if CUDA_VERSION >= 11000
    AddRelu<nv_bfloat16>(stream, n, reinterpret_cast<const nv_bfloat16*>(x),
                         reinterpret_cast<const nv_bfloat16*>(addend),
                         reinterpret_cast<nv_bfloat16*>(y), mask);
#else
    UNIMPLEMENTED();
#endif
  } else {
    UNIMPLEMENTED();
  }
}
void ReluBackward(ep::Stream* stream, int64_t n, const DataType data_type, const int32_t* mask,
                  const void* dy, void* addend_diff) {
  if (data_type == kFloat) {
    ReluBackward<float>(stream, n, mask, reinterpret_cast<const float*>(dy),
                        reinterpret_cast<float*>(addend_diff));
  } else if (data_type == kDouble) {
    ReluBackward<double>(stream, n, mask, reinterpret_cast<const double*>(dy),
                         reinterpret_cast<double*>(addend_diff));
  } else if (data_type == kFloat16) {
    ReluBackward<half>(stream, n, mask, reinterpret_cast<const half*>(dy),
                       reinterpret_cast<half*>(addend_diff));
  } else if (data_type == kBFloat16) {
#if CUDA_VERSION >= 11000
    ReluBackward<nv_bfloat16>(stream, n, mask, reinterpret_cast<const nv_bfloat16*>(dy),
                              reinterpret_cast<nv_bfloat16*>(addend_diff));
#else
    UNIMPLEMENTED();
#endif
  } else {
    UNIMPLEMENTED();
  }
}

class NormalizationTrainKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  NormalizationTrainKernel() = default;
  ~NormalizationTrainKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    if (ctx->op_type_name() == "normalization") { CHECK(ctx->Attr<bool>("training")); }
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto momentum = ctx->Attr<float>("momentum");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());
    cudnnBatchNormMode_t mode = getCudnnBatchNormMode(x->shape_view().NumAxes());
    const CudnnTensorDescHelper desc_helper(x->shape_view(), data_type, axis, mode);

    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(beta);
    desc_helper.CheckParamTensor(mean);
    desc_helper.CheckParamTensor(inv_variance);

    user_op::Tensor* moving_mean = nullptr;
    user_op::Tensor* moving_variance = nullptr;
    if (ctx->has_input("moving_mean", 0)) {
      CHECK(ctx->has_input("moving_variance", 0));
      moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
      moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
      desc_helper.CheckParamTensor(moving_mean);
      desc_helper.CheckParamTensor(moving_variance);
    }

    const void* sp_alpha = CudnnSPOnePtr(data_type);
    const void* sp_beta;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), y->data_type());
      CHECK_EQ(add_to_output->shape_view(), y->shape_view());
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), y->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      sp_beta = CudnnSPOnePtr(data_type);
    } else {
      sp_beta = CudnnSPZeroPtr(data_type);
    }

#if defined(BN_ENABLE_EX_API)
    size_t workspace_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, CUDNN_BATCHNORM_OPS_BN,
        desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(), desc_helper.param_desc(), nullptr,
        &workspace_size));
    size_t reserve_space_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, CUDNN_BATCHNORM_OPS_BN, nullptr,
        desc_helper.xy_desc(), &reserve_space_size));
    auto* workspace = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    if (reserve_space_size == 0 && workspace_size <= workspace->shape_view().elem_cnt()) {
      OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
          ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, CUDNN_BATCHNORM_OPS_BN,
          sp_alpha, sp_beta, desc_helper.xy_desc(), x->dptr(), nullptr, nullptr,
          desc_helper.xy_desc(), y->mut_dptr(), desc_helper.param_desc(), gamma->dptr(),
          beta->dptr(), 1.0 - momentum, moving_mean ? moving_mean->mut_dptr() : NULL,
          moving_variance ? moving_variance->mut_dptr() : NULL, epsilon, mean->mut_dptr(),
          inv_variance->mut_dptr(), nullptr, workspace->mut_dptr(),
          workspace->shape_view().elem_cnt(), nullptr, 0));
    } else {
      OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
          ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, sp_alpha, sp_beta,
          desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
          desc_helper.param_desc(), gamma->dptr(), beta->dptr(), 1.0 - momentum,
          moving_mean ? moving_mean->mut_dptr() : NULL,
          moving_variance ? moving_variance->mut_dptr() : NULL, epsilon, mean->mut_dptr(),
          inv_variance->mut_dptr()));
    }
#else
    OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, sp_alpha, sp_beta,
        desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
        desc_helper.param_desc(), gamma->dptr(), beta->dptr(), 1.0 - momentum,
        moving_mean ? moving_mean->mut_dptr() : NULL,
        moving_variance ? moving_variance->mut_dptr() : NULL, epsilon, mean->mut_dptr(),
        inv_variance->mut_dptr()));
#endif

    if (ctx->op_type_name() == "normalization_add_relu") {
      CHECK(!ctx->has_input("_add_to_output", 0));
      const int64_t elem_cnt = x->shape_view().elem_cnt();
      auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
      if (ctx->has_input("addend", 0)) {
        const auto* addend = ctx->Tensor4ArgNameAndIndex("addend", 0);
        AddRelu(ctx->stream(), elem_cnt, data_type, y->dptr(), addend->dptr(), y->mut_dptr(),
                mask->mut_dptr<int32_t>());
      } else {
        Relu(ctx->stream(), elem_cnt, data_type, y->dptr(), y->mut_dptr(),
             mask->mut_dptr<int32_t>());
      }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("normalization")
    .SetCreateFn<NormalizationTrainKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
                     && (user_op::HobAttr<bool>("training") == true))
    .SetInferTmpSizeFn(InferTrainTmpSize)
    .SetInplaceProposalFn([](const user_op::InferContext& ctx,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      if (ctx.has_input("_add_to_output", 0)) {
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));
      }
      return Maybe<void>::Ok();
    });

REGISTER_USER_KERNEL("normalization_add_relu")
    .SetCreateFn<NormalizationTrainKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA))
    .SetInferTmpSizeFn(InferTrainTmpSize);

class NormalizationGradUserKernel final : public user_op::OpKernel,
                                          public user_op::CudaGraphSupport {
 public:
  NormalizationGradUserKernel() = default;
  ~NormalizationGradUserKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
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
    CHECK_EQ(dy->shape_view(), x->shape_view());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape_view(), x->shape_view());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());
    cudnnBatchNormMode_t mode = getCudnnBatchNormMode(x->shape_view().NumAxes());
    const CudnnTensorDescHelper desc_helper(x->shape_view(), data_type, axis, mode);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(gamma_diff);
    desc_helper.CheckParamTensor(beta_diff);
    desc_helper.CheckParamTensor(mean);
    desc_helper.CheckParamTensor(inv_variance);

    void* bn_workspace_ptr;
    size_t bn_workspace_size;
    const void* bn_dy_ptr;

    if (ctx->op_type_name() == "normalization_grad") {
      bn_workspace_ptr = tmp_buffer->mut_dptr();
      bn_workspace_size = tmp_buffer->shape_view().elem_cnt();
      bn_dy_ptr = dy->dptr();
    } else if (ctx->op_type_name() == "normalization_add_relu_grad") {
      const int64_t elem_cnt = dy->shape_view().elem_cnt();
      const auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
      user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
      if (ctx->has_output("addend_diff", 0)) {
        user_op::Tensor* addend_diff = ctx->Tensor4ArgNameAndIndex("addend_diff", 0);
        ReluBackward(ctx->stream(), elem_cnt, data_type, mask->dptr<int32_t>(), dy->dptr(),
                     addend_diff->mut_dptr());
        bn_workspace_ptr = tmp_buffer->mut_dptr();
        bn_workspace_size = tmp_buffer->shape_view().elem_cnt();
        bn_dy_ptr = addend_diff->dptr();
      } else {
        const size_t tmp_buffer_size = tmp_buffer->shape_view().elem_cnt();
        const size_t relu_dx_size =
            GetCudaAlignedSize(dy->shape_view().elem_cnt() * GetSizeOfDataType(data_type));
        CHECK_GE(tmp_buffer_size, relu_dx_size);
        ReluBackward(ctx->stream(), elem_cnt, data_type, mask->dptr<int32_t>(), dy->dptr(),
                     tmp_buffer->mut_dptr());
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
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, CUDNN_BATCHNORM_OPS_BN,
        desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(), nullptr, desc_helper.xy_desc(),
        desc_helper.param_desc(), nullptr, &workspace_size));
    size_t reserve_space_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, CUDNN_BATCHNORM_OPS_BN, nullptr,
        desc_helper.xy_desc(), &reserve_space_size));
    if (reserve_space_size == 0 && workspace_size <= bn_workspace_size) {
      OF_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
          ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, CUDNN_BATCHNORM_OPS_BN,
          CudnnSPOnePtr(data_type), CudnnSPZeroPtr(data_type), CudnnSPOnePtr(data_type),
          CudnnSPZeroPtr(data_type), desc_helper.xy_desc(), x->dptr(), nullptr, nullptr,
          desc_helper.xy_desc(), bn_dy_ptr, nullptr, nullptr, desc_helper.xy_desc(), dx->mut_dptr(),
          desc_helper.param_desc(), gamma->dptr(), nullptr, gamma_diff->mut_dptr(),
          beta_diff->mut_dptr(), epsilon, mean->dptr(), inv_variance->dptr(), nullptr,
          bn_workspace_ptr, bn_workspace_size, nullptr, 0));
    } else {
      OF_CUDNN_CHECK(cudnnBatchNormalizationBackward(
          ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, CudnnSPOnePtr(data_type),
          CudnnSPZeroPtr(data_type), CudnnSPOnePtr(data_type), CudnnSPZeroPtr(data_type),
          desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), bn_dy_ptr, desc_helper.xy_desc(),
          dx->mut_dptr(), desc_helper.param_desc(), gamma->dptr(), gamma_diff->mut_dptr(),
          beta_diff->mut_dptr(), epsilon, mean->dptr(), inv_variance->dptr()));
    }
#else
    OF_CUDNN_CHECK(cudnnBatchNormalizationBackward(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), mode, CudnnSPOnePtr(data_type),
        CudnnSPZeroPtr(data_type), CudnnSPOnePtr(data_type), CudnnSPZeroPtr(data_type),
        desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), bn_dy_ptr, desc_helper.xy_desc(),
        dx->mut_dptr(), desc_helper.param_desc(), gamma->dptr(), gamma_diff->mut_dptr(),
        beta_diff->mut_dptr(), epsilon, mean->dptr(), inv_variance->dptr()));
#endif
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("normalization_grad")
    .SetCreateFn<NormalizationGradUserKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA))
    .SetInferTmpSizeFn(InferGradTmpSize);

#define REGISTER_BN_ADD_RELU_GRAD_KERNEL(dtype)
REGISTER_USER_KERNEL("normalization_add_relu_grad")
    .SetCreateFn<NormalizationGradUserKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA))
    .SetInferTmpSizeFn(InferGradTmpSize);

#if (CUDNN_VERSION >= 7401)

size_t InferFusedNormalizationAddReluTmpSize(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("x", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  const CudnnTensorDescHelper desc_helper(x.shape(), x.data_type(), axis,
                                          CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  size_t size_in_bytes;
  cudnnHandle_t handle = Singleton<CudnnHandlePool>::Get()->Get();
  CudnnActivationDesc activation_desc(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);
  cudnnBatchNormOps_t ops;
  cudnnTensorDescriptor_t z_desc;
  if (ctx->has_input("addend", 0)) {
    ops = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    z_desc = desc_helper.xy_desc();
  } else {
    ops = CUDNN_BATCHNORM_OPS_BN_ACTIVATION;
    z_desc = nullptr;
  }
  OF_CUDNN_CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
      handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, ops, desc_helper.xy_desc(), z_desc,
      desc_helper.xy_desc(), desc_helper.param_desc(), activation_desc.Get(), &size_in_bytes));
  Singleton<CudnnHandlePool>::Get()->Put(handle);
  return std::max(size_in_bytes, static_cast<size_t>(1));
}

size_t InferFusedNormalizationAddReluGradTmpSize(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("x", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  const CudnnTensorDescHelper desc_helper(x.shape(), x.data_type(), axis,
                                          CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
  size_t size_in_bytes;
  cudnnHandle_t handle = Singleton<CudnnHandlePool>::Get()->Get();
  CudnnActivationDesc activation_desc(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0);
  cudnnBatchNormOps_t ops;
  cudnnTensorDescriptor_t z_desc;
  if (ctx->has_output("addend_diff", 0)) {
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
  Singleton<CudnnHandlePool>::Get()->Put(handle);
  return std::max(size_in_bytes, static_cast<size_t>(1));
}

class FusedNormalizationAddReluKernel final : public user_op::OpKernel,
                                              public user_op::CudaGraphSupport {
 public:
  FusedNormalizationAddReluKernel() = default;
  ~FusedNormalizationAddReluKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
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
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

    const CudnnTensorDescHelper desc_helper(x->shape_view(), data_type, axis,
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
    if (ctx->has_input("addend", 0)) {
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
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        ops, desc_helper.xy_desc(), z_desc, desc_helper.xy_desc(), desc_helper.param_desc(),
        activation_desc.Get(), &min_workspace_size));
    const size_t workspace_size = tmp_buffer->shape_view().elem_cnt();
    CHECK_GE(workspace_size, min_workspace_size);
    size_t min_reserve_space_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        ops, activation_desc.Get(), desc_helper.xy_desc(), &min_reserve_space_size));
    const size_t reserve_space_size = reserve_space->shape_view().elem_cnt();
    CHECK_GE(reserve_space_size, min_reserve_space_size);

    OF_CUDNN_CHECK(cudnnBatchNormalizationForwardTrainingEx(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        ops, CudnnSPOnePtr(data_type), CudnnSPZeroPtr(data_type), desc_helper.xy_desc(), x->dptr(),
        z_desc, z_ptr, desc_helper.xy_desc(), y->mut_dptr(), desc_helper.param_desc(),
        gamma->dptr(), beta->dptr(), 1.0 - momentum, moving_mean->mut_dptr(),
        moving_variance->mut_dptr(), epsilon, mean->mut_dptr(), inv_variance->mut_dptr(),
        activation_desc.Get(), tmp_buffer->mut_dptr(), workspace_size, reserve_space->mut_dptr(),
        reserve_space_size));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("cudnn_fused_normalization_add_relu")
    .SetCreateFn<FusedNormalizationAddReluKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA))
    .SetInferTmpSizeFn(InferFusedNormalizationAddReluTmpSize);

class FusedNormalizationAddReluGradUserKernel final : public user_op::OpKernel,
                                                      public user_op::CudaGraphSupport {
 public:
  FusedNormalizationAddReluGradUserKernel() = default;
  ~FusedNormalizationAddReluGradUserKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
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
    CHECK_EQ(dy->shape_view(), x->shape_view());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape_view(), x->shape_view());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

    const CudnnTensorDescHelper desc_helper(x->shape_view(), data_type, axis,
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
    if (ctx->has_output("addend_diff", 0)) {
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
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        ops, desc_helper.xy_desc(), desc_helper.xy_desc(), desc_helper.xy_desc(), dz_desc,
        desc_helper.xy_desc(), desc_helper.param_desc(), activation_desc.Get(),
        &min_workspace_size));
    const size_t workspace_size = tmp_buffer->shape_view().elem_cnt();
    CHECK_GE(workspace_size, min_workspace_size);
    size_t min_reserve_space_size;
    OF_CUDNN_CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        ops, activation_desc.Get(), desc_helper.xy_desc(), &min_reserve_space_size));
    const size_t reserve_space_size = reserve_space->shape_view().elem_cnt();
    CHECK_GE(reserve_space_size, min_reserve_space_size);
    OF_CUDNN_CHECK(cudnnBatchNormalizationBackwardEx(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL_PERSISTENT,
        ops, CudnnSPOnePtr(data_type), CudnnSPZeroPtr(data_type), CudnnSPOnePtr(data_type),
        CudnnSPZeroPtr(data_type), desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(),
        y->dptr(), desc_helper.xy_desc(), dy->dptr(), dz_desc, dz_ptr, desc_helper.xy_desc(),
        dx->mut_dptr(), desc_helper.param_desc(), gamma->dptr(), beta->dptr(),
        gamma_diff->mut_dptr(), beta_diff->mut_dptr(), epsilon, mean->dptr(), inv_variance->dptr(),
        activation_desc.Get(), tmp_buffer->mut_dptr(), workspace_size,
        const_cast<void*>(reserve_space->dptr()), reserve_space_size));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("cudnn_fused_normalization_add_relu_grad")
    .SetCreateFn<FusedNormalizationAddReluGradUserKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA))
    .SetInferTmpSizeFn(InferFusedNormalizationAddReluGradTmpSize);

#endif

}  // namespace
}  // namespace oneflow

#endif
