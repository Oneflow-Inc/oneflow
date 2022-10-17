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
#include "oneflow/user/kernels/avg_pool_kernel_util.h"

namespace oneflow {

struct AvgPoolOpKernelCache final : public user_op::OpKernelCache {
  AvgPoolParams3D params_3d;
  explicit AvgPoolOpKernelCache(const AvgPoolParams3D& params_3d) : params_3d(params_3d) {}
  const AvgPoolParams3D& GetParams3D() const { return params_3d; }
};

std::shared_ptr<AvgPoolOpKernelCache> CreateAvgOpKernelCache(user_op::KernelCacheContext* ctx,
                                                             const int32_t& dim) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
  const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
  const bool ceil_mode = ctx->Attr<bool>("ceil_mode");
  const bool count_include_pad = ctx->Attr<bool>("count_include_pad");
  const int32_t divisor_override = ctx->Attr<int32_t>("divisor_override");

  AvgPoolParams3D params_3d =
      AvgPoolParams3D(dim, x_shape, data_format, padding, kernel_size, stride, ceil_mode,
                      count_include_pad, divisor_override);
  std::shared_ptr<AvgPoolOpKernelCache> cache(new AvgPoolOpKernelCache(params_3d));
  return cache;
}

template<typename T, typename IDX>
struct AvgPoolKernelUtil<DeviceType::kCPU, T, IDX> {
  static void Avgpool1dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d) {
    Avgpool1dForwardCompute<T, IDX>(
        index_helper, elem_num, src, dest, params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool1dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d) {
    Avgpool1dBackwardCompute<T, IDX>(
        index_helper, elem_num, src, dest, params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[2], params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool2dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d) {
    Avgpool2dForwardCompute<T, IDX>(
        index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.count_include_pad(),
        params_3d.divisor_override());
  }

  static void Avgpool2dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d) {
    Avgpool2dBackwardCompute<T, IDX>(
        index_helper, elem_num, src, dest, params_3d.padding()[1], params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.count_include_pad(),
        params_3d.divisor_override());
  }

  static void Avgpool3dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                               const IDX elem_num, const T* src, T* dest,
                               const AvgPoolParams3D& params_3d) {
    Avgpool3dForwardCompute<T, IDX>(
        index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
        params_3d.pool_size_3d()[0], params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[0], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.count_include_pad(), params_3d.divisor_override());
  }

  static void Avgpool3dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const AvgPoolParams3D& params_3d) {
    Avgpool3dBackwardCompute<T, IDX>(
        index_helper, elem_num, src, dest, params_3d.padding()[0], params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4),
        params_3d.pool_size_3d()[0], params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[0], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.count_include_pad(), params_3d.divisor_override());
  }
};

template<DeviceType device_type, typename T>
class AvgPool1dKernel final : public user_op::OpKernel {
 public:
  AvgPool1dKernel() = default;
  ~AvgPool1dKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateAvgOpKernelCache(ctx, 1);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto* pool_cache = dynamic_cast<const AvgPoolOpKernelCache*>(cache);
    const AvgPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = y->shape_view().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();

    DimVector y_vector(2);
    y_vector.at(0) = y->shape_view().At(0) * y->shape_view().At(1);
    y_vector.at(1) = y->shape_view().At(2);
    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(y_vector.data());
      AvgPoolKernelUtil<device_type, T, int32_t>::Avgpool1dForward(ctx->stream(), index_helper,
                                                                   elem_num, src, dest, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(y_vector.data());
      AvgPoolKernelUtil<device_type, T, int64_t>::Avgpool1dForward(ctx->stream(), index_helper,
                                                                   elem_num, src, dest, params_3d);
    }
  };
};

template<DeviceType device_type, typename T>
class AvgPool1dGradKernel final : public user_op::OpKernel {
 public:
  AvgPool1dGradKernel() = default;
  ~AvgPool1dGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateAvgOpKernelCache(ctx, 1);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pool_cache = dynamic_cast<const AvgPoolOpKernelCache*>(cache);
    const AvgPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = dy->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    T* dest = dx->mut_dptr<T>();
    size_t out_bytes_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    DimVector dy_vector(2);
    dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
    dy_vector.at(1) = dy->shape_view().At(2);
    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(dy_vector.data());
      AvgPoolKernelUtil<device_type, T, int32_t>::Avgpool1dBackward(ctx->stream(), index_helper,
                                                                    elem_num, src, dest, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(dy_vector.data());
      AvgPoolKernelUtil<device_type, T, int64_t>::Avgpool1dBackward(ctx->stream(), index_helper,
                                                                    elem_num, src, dest, params_3d);
    }
  };
};

template<DeviceType device_type, typename T>
class AvgPool2dKernel final : public user_op::OpKernel {
 public:
  AvgPool2dKernel() = default;
  ~AvgPool2dKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateAvgOpKernelCache(ctx, 2);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto* pool_cache = dynamic_cast<const AvgPoolOpKernelCache*>(cache);
    const AvgPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = y->shape_view().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();

    DimVector y_vector(3);
    y_vector.at(0) = y->shape_view().At(0) * y->shape_view().At(1);
    y_vector.at(1) = y->shape_view().At(2);
    y_vector.at(2) = y->shape_view().At(3);
    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 3> index_helper(y_vector.data());
      AvgPoolKernelUtil<device_type, T, int32_t>::Avgpool2dForward(ctx->stream(), index_helper,
                                                                   elem_num, src, dest, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 3> index_helper(y_vector.data());
      AvgPoolKernelUtil<device_type, T, int64_t>::Avgpool2dForward(ctx->stream(), index_helper,
                                                                   elem_num, src, dest, params_3d);
    }
  };
};

template<DeviceType device_type, typename T>
class AvgPool2dGradKernel final : public user_op::OpKernel {
 public:
  AvgPool2dGradKernel() = default;
  ~AvgPool2dGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateAvgOpKernelCache(ctx, 2);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pool_cache = dynamic_cast<const AvgPoolOpKernelCache*>(cache);
    const AvgPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = dy->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    T* dest = dx->mut_dptr<T>();

    size_t out_bytes_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    DimVector dy_vector(3);
    dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
    dy_vector.at(1) = dy->shape_view().At(2);
    dy_vector.at(2) = dy->shape_view().At(3);
    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 3> index_helper(dy_vector.data());
      AvgPoolKernelUtil<device_type, T, int32_t>::Avgpool2dBackward(ctx->stream(), index_helper,
                                                                    elem_num, src, dest, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 3> index_helper(dy_vector.data());
      AvgPoolKernelUtil<device_type, T, int64_t>::Avgpool2dBackward(ctx->stream(), index_helper,
                                                                    elem_num, src, dest, params_3d);
    }
  };
};

template<DeviceType device_type, typename T>
class AvgPool3dKernel final : public user_op::OpKernel {
 public:
  AvgPool3dKernel() = default;
  ~AvgPool3dKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateAvgOpKernelCache(ctx, 3);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto* pool_cache = dynamic_cast<const AvgPoolOpKernelCache*>(cache);
    const AvgPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = y->shape_view().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();

    DimVector y_vector(4);
    y_vector.at(0) = y->shape_view().At(0) * y->shape_view().At(1);
    y_vector.at(1) = y->shape_view().At(2);
    y_vector.at(2) = y->shape_view().At(3);
    y_vector.at(3) = y->shape_view().At(4);
    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 4> index_helper(y_vector.data());
      AvgPoolKernelUtil<device_type, T, int32_t>::Avgpool3dForward(ctx->stream(), index_helper,
                                                                   elem_num, src, dest, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());
      AvgPoolKernelUtil<device_type, T, int64_t>::Avgpool3dForward(ctx->stream(), index_helper,
                                                                   elem_num, src, dest, params_3d);
    }
  };
};

template<DeviceType device_type, typename T>
class AvgPool3dGradKernel final : public user_op::OpKernel {
 public:
  AvgPool3dGradKernel() = default;
  ~AvgPool3dGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateAvgOpKernelCache(ctx, 3);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pool_cache = dynamic_cast<const AvgPoolOpKernelCache*>(cache);
    const AvgPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = dy->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    T* dest = dx->mut_dptr<T>();

    size_t out_bytes_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    DimVector dy_vector(4);
    dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
    dy_vector.at(1) = dy->shape_view().At(2);
    dy_vector.at(2) = dy->shape_view().At(3);
    dy_vector.at(3) = dy->shape_view().At(4);
    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 4> index_helper(dy_vector.data());
      AvgPoolKernelUtil<device_type, T, int32_t>::Avgpool3dBackward(ctx->stream(), index_helper,
                                                                    elem_num, src, dest, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());
      AvgPoolKernelUtil<device_type, T, int64_t>::Avgpool3dBackward(ctx->stream(), index_helper,
                                                                    elem_num, src, dest, params_3d);
    }
  };
};

#define REGISTER_AVG_POOL_KERNELS(device, dtype)                                        \
  REGISTER_USER_KERNEL("avg_pool_1d")                                                   \
      .SetCreateFn<AvgPool1dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_1d_grad")                                              \
      .SetCreateFn<AvgPool1dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_2d")                                                   \
      .SetCreateFn<AvgPool2dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_2d_grad")                                              \
      .SetCreateFn<AvgPool2dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_3d")                                                   \
      .SetCreateFn<AvgPool3dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("avg_pool_3d_grad")                                              \
      .SetCreateFn<AvgPool3dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

#define REGISTER_AVG_POOL_WITH_DEVICE(device) \
  REGISTER_AVG_POOL_KERNELS(device, float)    \
  REGISTER_AVG_POOL_KERNELS(device, double)

REGISTER_AVG_POOL_WITH_DEVICE(DeviceType::kCPU)

#ifdef WITH_CUDA
REGISTER_AVG_POOL_WITH_DEVICE(DeviceType::kCUDA)
REGISTER_AVG_POOL_KERNELS(DeviceType::kCUDA, half)
#endif

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_AVG_POOL_KERNEL_UTIL, (DeviceType::kCPU),
                                 AVG_POOL_DATA_TYPE_CPU_SEQ, AVG_POOL_IDX_DATA_TYPE_SEQ);

}  // namespace oneflow
