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

struct PoolingOpKernelCache final : public user_op::OpKernelCache {
  MaxPoolingParams3D params_3d;
  explicit PoolingOpKernelCache(const MaxPoolingParams3D& params_3d) : params_3d(params_3d) {}
  const MaxPoolingParams3D& GetParams3D() const { return params_3d; }
};

std::shared_ptr<PoolingOpKernelCache> CreatePoolingOpKernelCache(user_op::KernelCacheContext* ctx,
                                                                 const int32_t& dim) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
  const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
  const std::vector<int32_t>& dilation = ctx->Attr<std::vector<int32_t>>("dilation");
  const bool return_indices = ctx->Attr<bool>("return_indices");
  const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

  MaxPoolingParams3D params_3d = MaxPoolingParams3D(dim, x_shape, data_format, padding, kernel_size,
                                                    stride, dilation, return_indices, ceil_mode);
  std::shared_ptr<PoolingOpKernelCache> cache(new PoolingOpKernelCache(params_3d));
  return cache;
}

namespace {

template<typename T>
void Maxpool2dForwardComputeCLast(const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                  int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                  const int32_t padding_h, const int32_t padding_w,
                                  const int64_t n_batch, const int64_t n_channel,
                                  const int64_t x_height, const int64_t x_width,
                                  const int64_t y_height, const int64_t y_width,
                                  const int32_t kernel_size_h, const int32_t kernel_size_w,
                                  const int32_t stride_h, const int32_t stride_w,
                                  const int32_t dilation_h, const int32_t dilation_w) {
  int64_t n = 0, h = 0, w = 0, c = 0;
  for (int64_t num = 0; num < elem_num; ++num) {
    index_helper.OffsetToNdIndex(num, n, h, w, c);

    const int64_t x_start_idx = n * x_height * x_width * n_channel;
    const int64_t y_start_idx = n * y_height * y_width * n_channel;
    int64_t hstart = h * stride_h - padding_h;
    int64_t wstart = w * stride_w - padding_w;
    const int64_t hend = (hstart + (kernel_size_h - 1) * dilation_h + 1) <= x_height
                             ? (hstart + (kernel_size_h - 1) * dilation_h + 1)
                             : x_height;
    const int64_t wend = (wstart + (kernel_size_w - 1) * dilation_w + 1) <= x_width
                             ? (wstart + (kernel_size_w - 1) * dilation_w + 1)
                             : x_width;

    while (hstart < 0) { hstart += dilation_h; }
    while (wstart < 0) { wstart += dilation_w; }
    /* compute max value(src[src_idx]) in kernel box region, and save the value to dest[num] */
    int64_t max_index = hstart * x_width + wstart;
    int64_t src_idx = 0;
    /* equal to -std::numeric_limits<T>::infinity(); */
    T max_value = detail::numeric_limits<T>::lower_bound();

    for (int64_t i = hstart; i < hend; i += dilation_h) {
      for (int64_t j = wstart; j < wend; j += dilation_w) {
        const int64_t window_idx = i * x_width * n_channel + j * n_channel + c;
        const int64_t search_idx = x_start_idx + window_idx;
        T val = src[search_idx];
        if (val > max_value || detail::numerics<T>::isnan(val)) {
          max_value = val;
          max_index = window_idx;
          src_idx = search_idx;
        }
      }
    }
    const int64_t out_idx = y_start_idx + h * y_width * n_channel + w * n_channel + c;
    dest[out_idx] = src[src_idx];
    indice_ptr[out_idx] = max_index;
  }
}

}  // namespace

template<typename T>
struct PoolingKernelUtil<DeviceType::kCPU, T> {
  static void Maxpool1dForward(ep::Stream* stream,
                               const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolingParams3D& params_3d) {
    Maxpool1dForwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr,
                               params_3d.padding()[2], params_3d.num_batch(),
                               params_3d.num_channel(), params_3d.GetXShape5D().At(4),
                               params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[2],
                               params_3d.stride_3d()[2], params_3d.dilation_3d()[2]);
  }

  static void Maxpool1dBackward(ep::Stream* stream,
                                const NdIndexOffsetHelper<int64_t, 3>& index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolingParams3D& params_3d) {
    Maxpool1dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr,
                                params_3d.num_batch(), params_3d.num_channel(),
                                params_3d.GetYShape5D().At(4), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool2dForwardCFirst(ep::Stream* stream,
                                     const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                     const int64_t elem_num, const T* src, T* dest,
                                     int64_t* indice_ptr, const MaxPoolingParams3D& params_3d) {
    Maxpool2dForwardComputeCFirst<T>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[1],
        params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackwardCFirst(ep::Stream* stream,
                                      const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                      const int64_t elem_num, const T* src, T* dest,
                                      const int64_t* indice_ptr,
                                      const MaxPoolingParams3D& params_3d) {
    Maxpool2dBackwardComputeCFirst<T>(index_helper, elem_num, src, dest, indice_ptr,
                                      params_3d.num_batch(), params_3d.num_channel(),
                                      params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
                                      params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool2dForwardCLast(ep::Stream* stream,
                                    const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                    const int64_t elem_num, const T* src, T* dest,
                                    int64_t* indice_ptr, const MaxPoolingParams3D& params_3d) {
    Maxpool2dForwardComputeCLast<T>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.pooling_size_3d()[1],
        params_3d.pooling_size_3d()[2], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackwardCLast(ep::Stream* stream,
                                     const NdIndexOffsetHelper<int64_t, 4>& index_helper,
                                     const int64_t elem_num, const T* src, T* dest,
                                     const int64_t* indice_ptr,
                                     const MaxPoolingParams3D& params_3d) {
    Maxpool2dBackwardComputeCLast<T>(index_helper, elem_num, src, dest, indice_ptr,
                                     params_3d.num_batch(), params_3d.num_channel(),
                                     params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
                                     params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool3dForward(ep::Stream* stream,
                               const NdIndexOffsetHelper<int64_t, 5>& index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolingParams3D& params_3d) {
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

  static void Maxpool3dBackward(ep::Stream* stream,
                                const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolingParams3D& params_3d) {
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

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreatePoolingOpKernelCache(ctx, 1);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto* pooling_cache = dynamic_cast<const PoolingOpKernelCache*>(cache);
    const MaxPoolingParams3D& params_3d = pooling_cache->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 3> index_helper(y_vector.data());

    PoolingKernelUtil<device_type, T>::Maxpool1dForward(ctx->stream(), index_helper, elem_num, src,
                                                        dest, indice_ptr, params_3d);
  };
};

template<DeviceType device_type, typename T>
class MaxPool1dGradKernel final : public user_op::OpKernel {
 public:
  MaxPool1dGradKernel() = default;
  ~MaxPool1dGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreatePoolingOpKernelCache(ctx, 1);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pooling_cache = dynamic_cast<const PoolingOpKernelCache*>(cache);
    const MaxPoolingParams3D& params_3d = pooling_cache->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 3> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    PoolingKernelUtil<device_type, T>::Maxpool1dBackward(ctx->stream(), index_helper, elem_num, src,
                                                         dest, indice_ptr, params_3d);
  };
};

template<DeviceType device_type, typename T>
class MaxPool2dKernel final : public user_op::OpKernel {
 public:
  MaxPool2dKernel() = default;
  ~MaxPool2dKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreatePoolingOpKernelCache(ctx, 2);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto* pooling_cache = dynamic_cast<const PoolingOpKernelCache*>(cache);
    const MaxPoolingParams3D& params_3d = pooling_cache->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      PoolingKernelUtil<device_type, T>::Maxpool2dForwardCFirst(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    } else if (data_format == "channels_last") {
      PoolingKernelUtil<device_type, T>::Maxpool2dForwardCLast(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    } else {
      UNIMPLEMENTED() << "Unsupported data_format";
    }
  };
};

template<DeviceType device_type, typename T>
class MaxPool2dGradKernel final : public user_op::OpKernel {
 public:
  MaxPool2dGradKernel() = default;
  ~MaxPool2dGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreatePoolingOpKernelCache(ctx, 2);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pooling_cache = dynamic_cast<const PoolingOpKernelCache*>(cache);
    const MaxPoolingParams3D& params_3d = pooling_cache->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      PoolingKernelUtil<device_type, T>::Maxpool2dBackwardCFirst(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    } else if (data_format == "channels_last") {
      PoolingKernelUtil<device_type, T>::Maxpool2dBackwardCLast(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    } else {
      UNIMPLEMENTED() << "Unsupported data_format";
    }
  };
};

template<DeviceType device_type, typename T>
class MaxPool3dKernel final : public user_op::OpKernel {
 public:
  MaxPool3dKernel() = default;
  ~MaxPool3dKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreatePoolingOpKernelCache(ctx, 3);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto* pooling_cache = dynamic_cast<const PoolingOpKernelCache*>(cache);
    const MaxPoolingParams3D& params_3d = pooling_cache->GetParams3D();

    const int64_t elem_num = y->shape().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    DimVector y_vector;
    y->shape().ToDimVector(&y_vector);
    NdIndexOffsetHelper<int64_t, 5> index_helper(y_vector.data());

    PoolingKernelUtil<device_type, T>::Maxpool3dForward(ctx->stream(), index_helper, elem_num, src,
                                                        dest, indice_ptr, params_3d);
  };
};

template<DeviceType device_type, typename T>
class MaxPool3dGradKernel final : public user_op::OpKernel {
 public:
  MaxPool3dGradKernel() = default;
  ~MaxPool3dGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreatePoolingOpKernelCache(ctx, 3);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pooling_cache = dynamic_cast<const PoolingOpKernelCache*>(cache);
    const MaxPoolingParams3D& params_3d = pooling_cache->GetParams3D();

    const int64_t elem_num = dy->shape().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();

    DimVector dy_vector;
    dy->shape().ToDimVector(&dy_vector);
    NdIndexOffsetHelper<int64_t, 5> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    PoolingKernelUtil<device_type, T>::Maxpool3dBackward(ctx->stream(), index_helper, elem_num, src,
                                                         dest, indice_ptr, params_3d);
  };
};

#define REGISTER_POOLING_KERNELS(device, dtype)                                         \
  REGISTER_USER_KERNEL("maxpool_1d")                                                    \
      .SetCreateFn<MaxPool1dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_1d_grad")                                               \
      .SetCreateFn<MaxPool1dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_2d")                                                    \
      .SetCreateFn<MaxPool2dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_2d_grad")                                               \
      .SetCreateFn<MaxPool2dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_3d")                                                    \
      .SetCreateFn<MaxPool3dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_3d_grad")                                               \
      .SetCreateFn<MaxPool3dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

#define REGISTER_POOLING_WITH_DEVICE(device) \
  REGISTER_POOLING_KERNELS(device, int32_t)  \
  REGISTER_POOLING_KERNELS(device, float)    \
  REGISTER_POOLING_KERNELS(device, double)

REGISTER_POOLING_WITH_DEVICE(DeviceType::kCPU)

#ifdef WITH_CUDA
REGISTER_POOLING_WITH_DEVICE(DeviceType::kCUDA)
// TODO: REGISTER_POOLING_KERNELS(DeviceType::kCUDA, float16)
#endif

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL_UTIL, (DeviceType::kCPU),
                                 POOLING_DATA_TYPE_CPU_SEQ);

}  // namespace oneflow
