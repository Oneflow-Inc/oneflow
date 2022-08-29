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
#ifdef WITH_ROCM

#include <unordered_map>
#include "hip/hip_runtime.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/rocm/cuda_stream.h"
#include "hip/hsa_detail/device_functions.h"

namespace oneflow {

namespace {

void InferDimSizeAndDataFormat(const ShapeView& x_shape, const int32_t axis, int32_t* n, int32_t* c,
                               int32_t* h, int32_t* w, hipdnnTensorFormat_t* format) {
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
    // *format = HIPDNN_TENSOR_NHWC;
    *format = HIPDNN_TENSOR_NCHW;
    // std::cout << "don't surpport HIPDNN_TENSOR_NHWC, use HIPDNN_TENSOR_NCHW instead, maybe cause wrong results" << std::endl;
  } else {
    *n = x_shape.Count(0, axis);
    *c = x_shape.At(axis);
    *h = x_shape.Count(axis + 1);
    *w = 1;
    *format = HIPDNN_TENSOR_NCHW;
  }
}

void InferXYCudnnTensorDesc(const ShapeView& xy_shape, const DataType& data_type,
                            const int32_t axis, hipdnnTensorDescriptor_t xy_desc) {
  int32_t n, c, h, w;
  hipdnnTensorFormat_t format;
  InferDimSizeAndDataFormat(xy_shape, axis, &n, &c, &h, &w, &format);
  OF_CUDNN_CHECK(
      hipdnnSetTensor4dDescriptor(xy_desc, format, GetCudnnDataType(data_type), n, c, h, w));
}

void InferParamCudnnTensorDesc(const hipdnnTensorDescriptor_t xy_desc, hipdnnBatchNormMode_t mode,
                               hipdnnTensorDescriptor_t param_desc) {
  OF_CUDNN_CHECK(hipdnnDeriveBNTensorDescriptor(param_desc, xy_desc, mode));
}

class CudnnTensorDescHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnTensorDescHelper);
  CudnnTensorDescHelper(const ShapeView& xy_shape, const DataType& data_type, const int32_t axis,
                        hipdnnBatchNormMode_t mode) {
    OF_CUDNN_CHECK(hipdnnCreateTensorDescriptor(&xy_desc_));
    InferXYCudnnTensorDesc(xy_shape, data_type, axis, xy_desc_);
    OF_CUDNN_CHECK(hipdnnCreateTensorDescriptor(&param_desc_));
    InferParamCudnnTensorDesc(xy_desc_, mode, param_desc_);
    int n, c, h, w, n_stride, c_stride, h_stride, w_stride;
    OF_CUDNN_CHECK(hipdnnGetTensor4dDescriptor(param_desc_, &param_data_type_, &n, &c, &h, &w,
                                              &n_stride, &c_stride, &h_stride, &w_stride));
    param_size_ = c;
  }
  ~CudnnTensorDescHelper() {
    OF_CUDNN_CHECK(hipdnnDestroyTensorDescriptor(param_desc_));
    OF_CUDNN_CHECK(hipdnnDestroyTensorDescriptor(xy_desc_));
  }

  hipdnnTensorDescriptor_t xy_desc() const { return xy_desc_; }

  hipdnnTensorDescriptor_t param_desc() const { return param_desc_; }

  void CheckParamTensor(const user_op::Tensor* tensor) const {
    CHECK_NOTNULL(tensor);
    CHECK_EQ(tensor->shape_view().NumAxes(), 1);
    CHECK_EQ(tensor->shape_view().At(0), param_size_);
    // CHECK_EQ(GetCudnnDataType(tensor->data_type()), param_data_type_);
  }

 private:
  hipdnnTensorDescriptor_t xy_desc_ = nullptr;
  hipdnnTensorDescriptor_t param_desc_ = nullptr;
  hipdnnDataType_t param_data_type_;
  int32_t param_size_ = 0;
};

size_t InferTrainWorkspaceSize(const ShapeView& x_shape, const DataType data_type,
                               const int32_t axis) {
  return 1;
}

size_t InferTrainTmpSize(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("x", 0);
  const auto axis = ctx->Attr<int32_t>("axis");
  return InferTrainWorkspaceSize(x.shape(), x.data_type(), axis);
}

size_t InferGradWorkspaceSize(const ShapeView& x_shape, const DataType data_type,
                              const int32_t axis) {
  return 1;
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

template<typename T>
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

    const CudnnTensorDescHelper desc_helper(x->shape_view(), data_type, axis,
                                            HIPDNN_BATCHNORM_SPATIAL);
    desc_helper.CheckParamTensor(gamma);
    desc_helper.CheckParamTensor(beta);
    desc_helper.CheckParamTensor(moving_mean);
    desc_helper.CheckParamTensor(moving_variance);

    const void* sp_alpha = CudnnSPOnePtr<T>();
    const void* sp_beta;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), y->data_type());
      CHECK_EQ(add_to_output->shape_view(), y->shape_view());
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), y->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      sp_beta = CudnnSPOnePtr<T>();
    } else {
      sp_beta = CudnnSPZeroPtr<T>();
    }

    OF_CUDNN_CHECK(hipdnnBatchNormalizationForwardInference(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), HIPDNN_BATCHNORM_SPATIAL, sp_alpha,
        sp_beta, desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
        desc_helper.param_desc(), gamma->dptr(), beta->dptr(), moving_mean->dptr(),
        moving_variance->dptr(), epsilon));
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_INFERENCE_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationInferenceKernel<dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)           \
                       && (user_op::HobAttr<bool>("training") == false))                        \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.has_input("_add_to_output", 0)) {                                               \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));           \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_BN_INFERENCE_KERNEL(float16)
REGISTER_BN_INFERENCE_KERNEL(float)
REGISTER_BN_INFERENCE_KERNEL(double)

#undef REGISTER_BN_INFERENCE_KERNEL

constexpr int64_t kCudaWarpSize = 64;

template<typename T>
__global__ void ReluGpu(int64_t n, const T* x, T* y, int64_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T x_val = x[i];
    const bool is_positive = (x_val > 0);
    int64_t warp_mask = __ballot(static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? x_val : 0;
  }
}

template<>
__global__ void ReluGpu<half>(int64_t n, const half* x, half* y, int64_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  const half zero = __float2half(0.0f);
  CUDA_1D_KERNEL_LOOP(i, n) {
    const half x_val = x[i];
    const bool is_positive = __hgt(x_val, zero);
    int64_t warp_mask = __ballot(static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? x_val : zero;
  }
}

template<typename T>
__global__ void AddReluGpu(int64_t n, const T* x, const T* addend, T* y, int64_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T sum = x[i] + addend[i];
    const bool is_positive = (sum > 0);
    int64_t warp_mask = __ballot(static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? sum : 0;
  }
}

template<>
__global__ void AddReluGpu<half>(int64_t n, const half* x, const half* addend, half* y,
                                 int64_t* mask) {
  const int32_t lane_id = threadIdx.x % kCudaWarpSize;
  const half zero = __float2half(0.0f);
  CUDA_1D_KERNEL_LOOP(i, n) {
    const half sum = __hadd(x[i], addend[i]);
    const bool is_positive = __hgt(sum, zero);
    int64_t warp_mask = __ballot(static_cast<int>(is_positive));
    if (lane_id == 0) { mask[i / kCudaWarpSize] = warp_mask; }
    y[i] = is_positive ? sum : zero;
  }
}

template<typename T>
void Relu(ep::Stream* stream, int64_t n, const T* x, T* y, int64_t* mask) {
  ReluGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
               stream->As<ep::CudaStream>()->cuda_stream()>>>(n, x, y, mask);
}

template<>
void Relu<float16>(ep::Stream* stream, int64_t n, const float16* x, float16* y, int64_t* mask) {
  Relu<half>(stream, n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y), mask);
}

template<typename T>
void AddRelu(ep::Stream* stream, int64_t n, const T* x, const T* addend, T* y, int64_t* mask) {
  AddReluGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                  stream->As<ep::CudaStream>()->cuda_stream()>>>(n, x, addend, y, mask);
}

template<>
void AddRelu<float16>(ep::Stream* stream, int64_t n, const float16* x, const float16* addend,
                      float16* y, int64_t* mask) {
  AddRelu<half>(stream, n, reinterpret_cast<const half*>(x), reinterpret_cast<const half*>(addend),
                reinterpret_cast<half*>(y), mask);
}

template<typename T>
__global__ void ReluBackwardGpu(int64_t n, const int64_t* mask, const T* dy, T* addend_diff) {
  int32_t lane_id = threadIdx.x % kCudaWarpSize;
  CUDA_1D_KERNEL_LOOP(i, n) {
    int64_t mask_val = mask[i / kCudaWarpSize];
    bool is_positive = mask_val & (1 << lane_id);
    addend_diff[i] = static_cast<T>(is_positive) * dy[i];
  }
}

template<typename T>
void ReluBackward(ep::Stream* stream, int64_t n, const int64_t* mask, const T* dy, T* addend_diff) {
  ReluBackwardGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                       stream->As<ep::CudaStream>()->cuda_stream()>>>(n, mask, dy, addend_diff);
}

template<>
void ReluBackward<float16>(ep::Stream* stream, int64_t n, const int64_t* mask, const float16* dy,
                           float16* addend_diff) {
  ReluBackward<half>(stream, n, mask, reinterpret_cast<const half*>(dy),
                     reinterpret_cast<half*>(addend_diff));
}

template<typename T>
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
    const CudnnTensorDescHelper desc_helper(x->shape_view(), data_type, axis,
                                            HIPDNN_BATCHNORM_SPATIAL_PERSISTENT);

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

    const void* sp_alpha = CudnnSPOnePtr<T>();
    const void* sp_beta;
    if (ctx->has_input("_add_to_output", 0)) {
      const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
      CHECK_EQ(add_to_output->data_type(), y->data_type());
      CHECK_EQ(add_to_output->shape_view(), y->shape_view());
      Memcpy<DeviceType::kCUDA>(
          ctx->stream(), y->mut_dptr<void>(), add_to_output->dptr<void>(),
          add_to_output->shape_view().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
      sp_beta = CudnnSPOnePtr<T>();
    } else {
      sp_beta = CudnnSPZeroPtr<T>();
    }

    OF_CUDNN_CHECK(hipdnnBatchNormalizationForwardTraining(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), HIPDNN_BATCHNORM_SPATIAL_PERSISTENT,
        const_cast<void *>(sp_alpha), const_cast<void *>(sp_beta), desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), y->mut_dptr(),
        desc_helper.param_desc(), const_cast<void *>(gamma->dptr()), const_cast<void *>(beta->dptr()), 1.0 - momentum,
        moving_mean ? moving_mean->mut_dptr() : NULL,
        moving_variance ? moving_variance->mut_dptr() : NULL, epsilon, mean->mut_dptr(),
        inv_variance->mut_dptr()));

    if (ctx->op_type_name() == "normalization_add_relu") {
      CHECK(!ctx->has_input("_add_to_output", 0));
      const int64_t elem_cnt = x->shape_view().elem_cnt();
      auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
      if (ctx->has_input("addend", 0)) {
        const auto* addend = ctx->Tensor4ArgNameAndIndex("addend", 0);
        AddRelu(ctx->stream(), elem_cnt, y->dptr<T>(), addend->dptr<T>(), y->mut_dptr<T>(),
                mask->mut_dptr<int64_t>());
      } else {
        Relu(ctx->stream(), elem_cnt, y->dptr<T>(), y->mut_dptr<T>(), mask->mut_dptr<int64_t>());
      }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_TRAIN_KERNEL(dtype)                                                         \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationTrainKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)           \
                       && (user_op::HobAttr<bool>("training") == true))                         \
      .SetInferTmpSizeFn(InferTrainTmpSize)                                                     \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.has_input("_add_to_output", 0)) {                                               \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));           \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_BN_TRAIN_KERNEL(float16)
REGISTER_BN_TRAIN_KERNEL(float)
REGISTER_BN_TRAIN_KERNEL(double)

#define REGISTER_BN_ADD_RELU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("normalization_add_relu")                                       \
      .SetCreateFn<NormalizationTrainKernel<dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferTrainTmpSize);

REGISTER_BN_ADD_RELU_KERNEL(float16)
REGISTER_BN_ADD_RELU_KERNEL(float)
REGISTER_BN_ADD_RELU_KERNEL(double)

template<typename T>
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

    const CudnnTensorDescHelper desc_helper(x->shape_view(), data_type, axis,
                                            HIPDNN_BATCHNORM_SPATIAL_PERSISTENT);
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
        ReluBackward(ctx->stream(), elem_cnt, mask->dptr<int64_t>(), dy->dptr<T>(),
                     addend_diff->mut_dptr<T>());
        bn_workspace_ptr = tmp_buffer->mut_dptr();
        bn_workspace_size = tmp_buffer->shape_view().elem_cnt();
        bn_dy_ptr = addend_diff->dptr();
      } else {
        const size_t tmp_buffer_size = tmp_buffer->shape_view().elem_cnt();
        const size_t relu_dx_size =
            GetCudaAlignedSize(dy->shape_view().elem_cnt() * GetSizeOfDataType(dy->data_type()));
        CHECK_GE(tmp_buffer_size, relu_dx_size);
        ReluBackward(ctx->stream(), elem_cnt, mask->dptr<int64_t>(), dy->dptr<T>(),
                     reinterpret_cast<T*>(tmp_buffer->mut_dptr()));
        bn_workspace_ptr = tmp_buffer->mut_dptr<char>() + relu_dx_size;
        bn_workspace_size = tmp_buffer_size - relu_dx_size;
        bn_dy_ptr = tmp_buffer->dptr();
      }
    } else {
      UNIMPLEMENTED();
    }

    OF_CUDNN_CHECK(hipdnnBatchNormalizationBackward(
        ctx->stream()->As<ep::CudaStream>()->cudnn_handle(), HIPDNN_BATCHNORM_SPATIAL_PERSISTENT,
        CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(),
        desc_helper.xy_desc(), x->dptr(), desc_helper.xy_desc(), bn_dy_ptr, desc_helper.xy_desc(),
        dx->mut_dptr(), desc_helper.param_desc(), gamma->dptr(), gamma_diff->mut_dptr(),
        beta_diff->mut_dptr(), epsilon, mean->dptr(), inv_variance->dptr()));

  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_GRAD_KERNEL(dtype)                                                  \
  REGISTER_USER_KERNEL("normalization_grad")                                            \
      .SetCreateFn<NormalizationGradUserKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferGradTmpSize);

REGISTER_BN_GRAD_KERNEL(float16)
REGISTER_BN_GRAD_KERNEL(float)
REGISTER_BN_GRAD_KERNEL(double)

#define REGISTER_BN_ADD_RELU_GRAD_KERNEL(dtype)                                         \
  REGISTER_USER_KERNEL("normalization_add_relu_grad")                                   \
      .SetCreateFn<NormalizationGradUserKernel<dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferGradTmpSize);

REGISTER_BN_ADD_RELU_GRAD_KERNEL(float16)
REGISTER_BN_ADD_RELU_GRAD_KERNEL(float)
REGISTER_BN_ADD_RELU_GRAD_KERNEL(double)


}  // namespace
}  // namespace oneflow

#endif