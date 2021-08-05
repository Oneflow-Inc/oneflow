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
#include "oneflow/user/kernels/avg_pooling_kernel_util.h"

namespace oneflow {

struct AvgPoolingOpKernelState final : public user_op::OpKernelState {
  AvgPoolingParams3D params_3d;
  AvgPoolingOpKernelState(AvgPoolingParams3D params_3d) : params_3d(params_3d) {}
  const AvgPoolingParams3D& GetParams3D() { return params_3d; }
};

std::shared_ptr<AvgPoolingOpKernelState> DoCreateAvgOpKernelState(
    user_op::KernelComputeContext* ctx, const int32_t& dim) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
  const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
  const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
  const bool count_include_pad = ctx->Attr<bool>("count_include_pad");
  const int64_t divisor_override = ctx->Attr<int64_t>("divisor_override");

  AvgPoolingParams3D params_3d =
      AvgPoolingParams3D(dim, x_shape, data_format, padding, kernel_size, stride, ceil_mode,
                         count_include_pad, divisor_override);
  std::shared_ptr<AvgPoolingOpKernelState> state(new AvgPoolingOpKernelState(params_3d));
  return state;
}

template<typename T>
struct AvgPoolingKernelUtil<DeviceType::kCPU, T> {
  static void Avgpool1dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d) {
    Avgpool1dForwardCompute<T>(index_helper, elem_num, src, dest, params_3d.padding()[2],
                               params_3d.num_batch(), params_3d.num_channel(),
                               params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(4),
                               params_3d.pooling_size_3d()[2], params_3d.stride_3d()[2],
                               params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool1dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d) {
    Avgpool1dBackwardCompute<T>(index_helper, elem_num, src, dest, params_3d.padding()[2],
                                params_3d.num_batch(), params_3d.num_channel(),
                                params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(4),
                                params_3d.pooling_size_3d()[2], params_3d.stride_3d()[2],
                                params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d) {
    Avgpool2dForwardCompute<T>(
        index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
        params_3d.pooling_size_3d()[1], params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d) {
    Avgpool2dBackwardCompute<T>(
        index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
        params_3d.pooling_size_3d()[1], params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool3dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                               const int64_t elem_num, const T* src, T* dest,
                               const AvgPoolingParams3D& params_3d) {
    Avgpool3dForwardCompute<T>(
        index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
        params_3d.GetYShape5D().At(2), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
        params_3d.pooling_size_3d()[0], params_3d.pooling_size_3d()[1],
        params_3d.pooling_size_3d()[2], params_3d.stride_3d()[0], params_3d.stride_3d()[1],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolingParams3D& params_3d) {
    Avgpool3dBackwardCompute<T>(
        index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
        params_3d.GetYShape5D().At(2), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
        params_3d.pooling_size_3d()[0], params_3d.pooling_size_3d()[1],
        params_3d.pooling_size_3d()[2], params_3d.stride_3d()[0], params_3d.stride_3d()[1],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }
};

template<DeviceType device_type, typename T>
class AvgPool1dKernel final : public user_op::OpKernel {
 public:
  AvgPool1dKernel() = default;
  ~AvgPool1dKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto& pooling_state = DoCreateAvgOpKernelState(ctx, 1);
    const AvgPoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 3> index_helper(y_vector.data());
    AvgPoolingKernelUtil<device_type, T>::Avgpool1dForward(ctx->device_ctx(), index_helper,
                                                           elem_num, src, dest, params_3d);
  };
};

template<DeviceType device_type, typename T>
class AvgPool1dGradKernel final : public user_op::OpKernel {
 public:
  AvgPool1dGradKernel() = default;
  ~AvgPool1dGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto& pooling_state = DoCreateAvgOpKernelState(ctx, 1);
    const AvgPoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 3> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);
    AvgPoolingKernelUtil<device_type, T>::Avgpool1dBackward(ctx->device_ctx(), index_helper,
                                                            elem_num, src, dest, params_3d);
  };
};

template<DeviceType device_type, typename T>
class AvgPool2dKernel final : public user_op::OpKernel {
 public:
  AvgPool2dKernel() = default;
  ~AvgPool2dKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto& pooling_state = DoCreateAvgOpKernelState(ctx, 2);
    const AvgPoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());
    AvgPoolingKernelUtil<device_type, T>::Avgpool2dForward(ctx->device_ctx(), index_helper,
                                                           elem_num, src, dest, params_3d);
  };
};

template<DeviceType device_type, typename T>
class AvgPool2dGradKernel final : public user_op::OpKernel {
 public:
  AvgPool2dGradKernel() = default;
  ~AvgPool2dGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto& pooling_state = DoCreateAvgOpKernelState(ctx, 2);
    const AvgPoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);
    AvgPoolingKernelUtil<device_type, T>::Avgpool2dBackward(ctx->device_ctx(), index_helper,
                                                            elem_num, src, dest, params_3d);
  };
};

template<DeviceType device_type, typename T>
class AvgPool3dKernel final : public user_op::OpKernel {
 public:
  AvgPool3dKernel() = default;
  ~AvgPool3dKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto& pooling_state = DoCreateAvgOpKernelState(ctx, 3);
    const AvgPoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 5> index_helper(y_vector.data());
    AvgPoolingKernelUtil<device_type, T>::Avgpool3dForward(ctx->device_ctx(), index_helper,
                                                           elem_num, src, dest, params_3d);
  };
};

template<DeviceType device_type, typename T>
class AvgPool3dGradKernel final : public user_op::OpKernel {
 public:
  AvgPool3dGradKernel() = default;
  ~AvgPool3dGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto& pooling_state = DoCreateAvgOpKernelState(ctx, 3);
    const AvgPoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 5> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);
    AvgPoolingKernelUtil<device_type, T>::Avgpool3dBackward(ctx->device_ctx(), index_helper,
                                                            elem_num, src, dest, params_3d);
  };
};

#define REGISTER_AVG_POOLING_KERNELS(device, dtype)                                    \
  REGISTER_USER_KERNEL("avgpool_1d")                                                   \
      .SetCreateFn<AvgPool1dKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avgpool_1d_grad")                                              \
      .SetCreateFn<AvgPool1dGradKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avgpool_2d")                                                   \
      .SetCreateFn<AvgPool2dKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avgpool_2d_grad")                                              \
      .SetCreateFn<AvgPool2dGradKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avgpool_3d")                                                   \
      .SetCreateFn<AvgPool3dKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avgpool_3d_grad")                                              \
      .SetCreateFn<AvgPool3dGradKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

#define REGISTER_AVG_POOLING_WITH_DEVICE(device) \
  REGISTER_AVG_POOLING_KERNELS(device, float)    \
  REGISTER_AVG_POOLING_KERNELS(device, double)

REGISTER_AVG_POOLING_WITH_DEVICE(DeviceType::kCPU)

#ifdef WITH_CUDA
REGISTER_AVG_POOLING_WITH_DEVICE(DeviceType::kGPU)
// TODO: REGISTER_POOLING_KERNELS(DeviceType::kGPU, float16)
#endif

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_AVG_POOLING_KERNEL_UTIL, (DeviceType::kCPU),
                                 AVG_POOLING_DATA_TYPE_CPU_SEQ);

}  // namespace oneflow