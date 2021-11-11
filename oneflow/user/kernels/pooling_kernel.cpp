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
#include "oneflow/user/kernels/pooling_kernel_util.h"

namespace oneflow {

struct PoolingOpKernelState final : public user_op::OpKernelState {
  PoolingParams3D params_3d;
  PoolingOpKernelState(PoolingParams3D params_3d) : params_3d(params_3d) {}
  const PoolingParams3D& GetParams3D() { return params_3d; }
};

std::shared_ptr<PoolingOpKernelState> DoCreateOpKernelState(user_op::KernelComputeContext* ctx,
                                                            const int32_t& dim) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
  const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
  const std::vector<int32_t>& dilation = ctx->Attr<std::vector<int32_t>>("dilation");
  const bool return_indices = ctx->Attr<bool>("return_indices");
  const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

  PoolingParams3D params_3d = PoolingParams3D(dim, x_shape, data_format, padding, kernel_size,
                                              stride, dilation, return_indices, ceil_mode);
  std::shared_ptr<PoolingOpKernelState> state(new PoolingOpKernelState(params_3d));
  return state;
}

template<typename T>
struct PoolingKernelUtil<DeviceType::kCPU, T> {
  static void Maxpool1dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const PoolingParams3D& params_3d) {
    Maxpool1dForwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr,
                               params_3d.padding()[2], params_3d.num_batch(),
                               params_3d.num_channel(), params_3d.GetXShape5D().At(4),
                               params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[2],
                               params_3d.stride_3d()[2], params_3d.dilation_3d()[2]);
  }

  static void Maxpool1dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const PoolingParams3D& params_3d) {
    Maxpool1dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr,
                                params_3d.num_batch(), params_3d.num_channel(),
                                params_3d.GetYShape5D().At(4), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const PoolingParams3D& params_3d) {
    Maxpool2dForwardCompute<T>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[1],
        params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const PoolingParams3D& params_3d) {
    Maxpool2dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr,
                                params_3d.num_batch(), params_3d.num_channel(),
                                params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
                                params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool3dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const PoolingParams3D& params_3d) {
    Maxpool3dForwardCompute<T>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[0],
        params_3d.padding()[1], params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(2), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[0],
        params_3d.pooling_size_3d()[1], params_3d.pooling_size_3d()[2], params_3d.stride_3d()[0],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.dilation_3d()[0],
        params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const PoolingParams3D& params_3d) {
    Maxpool3dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr,
                                params_3d.num_batch(), params_3d.num_channel(),
                                params_3d.GetYShape5D().At(2), params_3d.GetYShape5D().At(3),
                                params_3d.GetYShape5D().At(4), params_3d.GetXShape5D().At(2),
                                params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }
};

template<DeviceType device_type, typename T>
class MaxPool1dKernel final : public user_op::OpKernel {
 public:
  MaxPool1dKernel() = default;
  ~MaxPool1dKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto& pooling_state = DoCreateOpKernelState(ctx, 1);
    const PoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 3> index_helper(y_vector.data());

    PoolingKernelUtil<device_type, T>::Maxpool1dForward(ctx->device_ctx(), index_helper, elem_num,
                                                        src, dest, indice_ptr, params_3d);
  };
};

template<DeviceType device_type, typename T>
class MaxPool1dGradKernel final : public user_op::OpKernel {
 public:
  MaxPool1dGradKernel() = default;
  ~MaxPool1dGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto& pooling_state = DoCreateOpKernelState(ctx, 1);
    const PoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 3> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);

    PoolingKernelUtil<device_type, T>::Maxpool1dBackward(ctx->device_ctx(), index_helper, elem_num,
                                                         src, dest, indice_ptr, params_3d);
  };
};

template<DeviceType device_type, typename T>
class MaxPool2dKernel final : public user_op::OpKernel {
 public:
  MaxPool2dKernel() = default;
  ~MaxPool2dKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto& pooling_state = DoCreateOpKernelState(ctx, 2);
    const PoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());

    PoolingKernelUtil<device_type, T>::Maxpool2dForward(ctx->device_ctx(), index_helper, elem_num,
                                                        src, dest, indice_ptr, params_3d);
  };
};

template<DeviceType device_type, typename T>
class MaxPool2dGradKernel final : public user_op::OpKernel {
 public:
  MaxPool2dGradKernel() = default;
  ~MaxPool2dGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto& pooling_state = DoCreateOpKernelState(ctx, 2);
    const PoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);

    PoolingKernelUtil<device_type, T>::Maxpool2dBackward(ctx->device_ctx(), index_helper, elem_num,
                                                         src, dest, indice_ptr, params_3d);
  };
};

template<DeviceType device_type, typename T>
class MaxPool3dKernel final : public user_op::OpKernel {
 public:
  MaxPool3dKernel() = default;
  ~MaxPool3dKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto& pooling_state = DoCreateOpKernelState(ctx, 3);
    const PoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 5> index_helper(y_vector.data());

    PoolingKernelUtil<device_type, T>::Maxpool3dForward(ctx->device_ctx(), index_helper, elem_num,
                                                        src, dest, indice_ptr, params_3d);
  };
};

template<DeviceType device_type, typename T>
class MaxPool3dGradKernel final : public user_op::OpKernel {
 public:
  MaxPool3dGradKernel() = default;
  ~MaxPool3dGradKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto& pooling_state = DoCreateOpKernelState(ctx, 3);
    const PoolingParams3D& params_3d = pooling_state->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();

    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 5> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);

    PoolingKernelUtil<device_type, T>::Maxpool3dBackward(ctx->device_ctx(), index_helper, elem_num,
                                                         src, dest, indice_ptr, params_3d);
  };
};

#define REGISTER_POOLING_KERNELS(device, dtype)                                        \
  REGISTER_USER_KERNEL("maxpool_1d")                                                   \
      .SetCreateFn<MaxPool1dKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                            \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_1d_grad")                                              \
      .SetCreateFn<MaxPool1dGradKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                            \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_2d")                                                   \
      .SetCreateFn<MaxPool2dKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                            \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_2d_grad")                                              \
      .SetCreateFn<MaxPool2dGradKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                            \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_3d")                                                   \
      .SetCreateFn<MaxPool3dKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                            \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_3d_grad")                                              \
      .SetCreateFn<MaxPool3dGradKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                            \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

#define REGISTER_POOLING_WITH_DEVICE(device) \
  REGISTER_POOLING_KERNELS(device, int32_t)  \
  REGISTER_POOLING_KERNELS(device, float)    \
  REGISTER_POOLING_KERNELS(device, double)

REGISTER_POOLING_WITH_DEVICE(DeviceType::kCPU)

#ifdef WITH_CUDA
REGISTER_POOLING_WITH_DEVICE(DeviceType::kGPU)
// TODO: REGISTER_POOLING_KERNELS(DeviceType::kGPU, float16)
#endif

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL_UTIL, (DeviceType::kCPU),
                                 POOLING_DATA_TYPE_CPU_SEQ);

}  // namespace oneflow
