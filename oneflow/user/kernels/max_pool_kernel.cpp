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
#include "oneflow/user/kernels/max_pool_kernel_util.h"

namespace oneflow {

struct PoolOpKernelCache final : public user_op::OpKernelCache {
  MaxPoolParams3D params_3d;
  explicit PoolOpKernelCache(const MaxPoolParams3D& params_3d) : params_3d(params_3d) {}
  const MaxPoolParams3D& GetParams3D() const { return params_3d; }
};

std::shared_ptr<PoolOpKernelCache> CreatePoolOpKernelCache(user_op::KernelCacheContext* ctx,
                                                           const int32_t& dim) {
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const std::string& data_format = ctx->Attr<std::string>("data_format");
  const std::vector<int32_t>& padding = ctx->Attr<std::vector<int32_t>>("padding");
  const std::vector<int32_t>& kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
  const std::vector<int32_t>& stride = ctx->Attr<std::vector<int32_t>>("stride");
  const std::vector<int32_t>& dilation = ctx->Attr<std::vector<int32_t>>("dilation");
  const bool return_indices = ctx->Attr<bool>("return_indices");
  const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

  MaxPoolParams3D params_3d = MaxPoolParams3D(dim, x_shape, data_format, padding, kernel_size,
                                              stride, dilation, return_indices, ceil_mode);
  std::shared_ptr<PoolOpKernelCache> cache(new PoolOpKernelCache(params_3d));
  return cache;
}

namespace {

template<typename T, typename IDX>
void Maxpool2dForwardComputeCLast(const NdIndexOffsetHelper<IDX, 4>& index_helper, IDX elem_num,
                                  const T* src, T* dest, int64_t* indice_ptr,
                                  const int32_t padding_h, const int32_t padding_w,
                                  const int32_t n_batch, const int32_t n_channel,
                                  const int32_t x_height, const int32_t x_width,
                                  const int32_t y_height, const int32_t y_width,
                                  const int32_t kernel_size_h, const int32_t kernel_size_w,
                                  const int32_t stride_h, const int32_t stride_w,
                                  const int32_t dilation_h, const int32_t dilation_w) {
  IDX n = 0, h = 0, w = 0, c = 0;
  for (IDX num = 0; num < elem_num; ++num) {
    index_helper.OffsetToNdIndex(num, n, h, w, c);

    const IDX x_start_idx = n * x_height * x_width * n_channel;
    const IDX y_start_idx = n * y_height * y_width * n_channel;
    IDX hstart = h * stride_h - padding_h;
    IDX wstart = w * stride_w - padding_w;
    const IDX hend = (hstart + (kernel_size_h - 1) * dilation_h + 1) <= x_height
                         ? (hstart + (kernel_size_h - 1) * dilation_h + 1)
                         : x_height;
    const IDX wend = (wstart + (kernel_size_w - 1) * dilation_w + 1) <= x_width
                         ? (wstart + (kernel_size_w - 1) * dilation_w + 1)
                         : x_width;

    while (hstart < 0) { hstart += dilation_h; }
    while (wstart < 0) { wstart += dilation_w; }
    /* compute max value(src[src_idx]) in kernel box region, and save the value to dest[num] */
    IDX max_index = hstart * x_width + wstart;
    IDX src_idx = 0;
    /* equal to -std::numeric_limits<T>::infinity(); */
    T max_value = detail::numeric_limits<T>::lower_bound();

    for (IDX i = hstart; i < hend; i += dilation_h) {
      for (IDX j = wstart; j < wend; j += dilation_w) {
        const IDX window_idx = i * x_width * n_channel + j * n_channel + c;
        const IDX search_idx = x_start_idx + window_idx;
        T val = src[search_idx];
        if (val > max_value || detail::numerics<T>::isnan(val)) {
          max_value = val;
          max_index = window_idx;
          src_idx = search_idx;
        }
      }
    }
    const IDX out_idx = y_start_idx + h * y_width * n_channel + w * n_channel + c;
    dest[out_idx] = src[src_idx];
    indice_ptr[out_idx] = max_index;
  }
}

}  // namespace

template<typename T, typename IDX>
struct PoolKernelUtil<DeviceType::kCPU, T, IDX> {
  static void Maxpool1dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                               const IDX elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolParams3D& params_3d) {
    Maxpool1dForwardCompute<T, IDX>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[2],
        params_3d.num_batch(), params_3d.num_channel(), params_3d.GetXShape5D().At(4),
        params_3d.pool_size_3d()[2], params_3d.stride_3d()[2], params_3d.dilation_3d()[2]);
  }

  static void Maxpool1dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 2>& index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolParams3D& params_3d) {
    Maxpool1dBackwardCompute<T, IDX>(index_helper, elem_num, src, dest, indice_ptr,
                                     params_3d.num_batch(), params_3d.num_channel(),
                                     params_3d.GetYShape5D().At(4), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool2dForwardCFirst(ep::Stream* stream,
                                     const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                     const IDX elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                     const MaxPoolParams3D& params_3d) {
    Maxpool2dForwardComputeCFirst<T, IDX>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[1],
        params_3d.pool_size_3d()[2], params_3d.stride_3d()[1], params_3d.stride_3d()[2],
        params_3d.dilation_3d()[1], params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackwardCFirst(ep::Stream* stream,
                                      const NdIndexOffsetHelper<IDX, 3>& index_helper,
                                      const IDX elem_num, const T* src, T* dest,
                                      const int64_t* indice_ptr, const MaxPoolParams3D& params_3d) {
    Maxpool2dBackwardComputeCFirst<T, IDX>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool2dForwardCLast(ep::Stream* stream,
                                    const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                    const IDX elem_num, const T* src, T* dest, int64_t* indice_ptr,
                                    const MaxPoolParams3D& params_3d) {
    Maxpool2dForwardComputeCLast<T, IDX>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[1],
        params_3d.padding()[2], params_3d.num_batch(), params_3d.num_channel(),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4), params_3d.GetYShape5D().At(3),
        params_3d.GetYShape5D().At(4), params_3d.pool_size_3d()[1], params_3d.pool_size_3d()[2],
        params_3d.stride_3d()[1], params_3d.stride_3d()[2], params_3d.dilation_3d()[1],
        params_3d.dilation_3d()[2]);
  }

  static void Maxpool2dBackwardCLast(ep::Stream* stream,
                                     const NdIndexOffsetHelper<IDX, 4>& index_helper,
                                     const IDX elem_num, const T* src, T* dest,
                                     const int64_t* indice_ptr, const MaxPoolParams3D& params_3d) {
    Maxpool2dBackwardComputeCLast<T, IDX>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetYShape5D().At(3), params_3d.GetYShape5D().At(4),
        params_3d.GetXShape5D().At(3), params_3d.GetXShape5D().At(4));
  }

  static void Maxpool3dForward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4>& index_helper,
                               const IDX elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const MaxPoolParams3D& params_3d) {
    Maxpool3dForwardCompute<T, IDX>(
        index_helper, elem_num, src, dest, indice_ptr, params_3d.padding()[0],
        params_3d.padding()[1], params_3d.padding()[2], params_3d.num_batch(),
        params_3d.num_channel(), params_3d.GetXShape5D().At(2), params_3d.GetXShape5D().At(3),
        params_3d.GetXShape5D().At(4), params_3d.pool_size_3d()[0], params_3d.pool_size_3d()[1],
        params_3d.pool_size_3d()[2], params_3d.stride_3d()[0], params_3d.stride_3d()[1],
        params_3d.stride_3d()[2], params_3d.dilation_3d()[0], params_3d.dilation_3d()[1],
        params_3d.dilation_3d()[2]);
  }

  static void Maxpool3dBackward(ep::Stream* stream, const NdIndexOffsetHelper<IDX, 4> index_helper,
                                const IDX elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const MaxPoolParams3D& params_3d) {
    Maxpool3dBackwardCompute<T, IDX>(index_helper, elem_num, src, dest, indice_ptr,
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
    return CreatePoolOpKernelCache(ctx, 1);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
    const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = y->shape_view().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    DimVector y_vector(2);
    y_vector.at(0) = y->shape_view().At(0) * y->shape_view().At(1);
    y_vector.at(1) = y->shape_view().At(2);
    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(y_vector.data());
      PoolKernelUtil<device_type, T, int32_t>::Maxpool1dForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(y_vector.data());
      PoolKernelUtil<device_type, T, int64_t>::Maxpool1dForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    }
  }
};

template<DeviceType device_type, typename T>
class MaxPool1dGradKernel final : public user_op::OpKernel {
 public:
  MaxPool1dGradKernel() = default;
  ~MaxPool1dGradKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreatePoolOpKernelCache(ctx, 1);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
    const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = dy->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector(2);
    dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
    dy_vector.at(1) = dy->shape_view().At(2);
    size_t out_bytes_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 2> index_helper(dy_vector.data());
      PoolKernelUtil<device_type, T, int32_t>::Maxpool1dBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 2> index_helper(dy_vector.data());
      PoolKernelUtil<device_type, T, int64_t>::Maxpool1dBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    }
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
    return CreatePoolOpKernelCache(ctx, 2);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
    const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = y->shape_view().elem_cnt();

    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    const std::string& data_format = ctx->Attr<std::string>("data_format");
    if (data_format == "channels_first") {
      DimVector y_vector(3);
      y_vector.at(0) = y->shape_view().At(0) * y->shape_view().At(1);
      y_vector.at(1) = y->shape_view().At(2);
      y_vector.at(2) = y->shape_view().At(3);
      if (elem_num < GetMaxVal<int32_t>()) {
        NdIndexOffsetHelper<int32_t, 3> index_helper(y_vector.data());
        PoolKernelUtil<device_type, T, int32_t>::Maxpool2dForwardCFirst(
            ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
      } else {
        NdIndexOffsetHelper<int64_t, 3> index_helper(y_vector.data());
        PoolKernelUtil<device_type, T, int64_t>::Maxpool2dForwardCFirst(
            ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
      }
    } else if (data_format == "channels_last") {
      DimVector y_vector;
      y->shape_view().ToDimVector(&y_vector);
      if (elem_num < GetMaxVal<int32_t>()) {
        NdIndexOffsetHelper<int32_t, 4> index_helper(y_vector.data());
        PoolKernelUtil<device_type, T, int32_t>::Maxpool2dForwardCLast(
            ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
      } else {
        NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());
        PoolKernelUtil<device_type, T, int64_t>::Maxpool2dForwardCLast(
            ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
      }
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
    return CreatePoolOpKernelCache(ctx, 2);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
    const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = dy->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();

    size_t out_bytes_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    const std::string& data_format = ctx->Attr<std::string>("data_format");

    if (data_format == "channels_first") {
      DimVector dy_vector(3);
      dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
      dy_vector.at(1) = dy->shape_view().At(2);
      dy_vector.at(2) = dy->shape_view().At(3);
      if (elem_num < GetMaxVal<int32_t>()) {
        NdIndexOffsetHelper<int32_t, 3> index_helper(dy_vector.data());
        PoolKernelUtil<device_type, T, int32_t>::Maxpool2dBackwardCFirst(
            ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
      } else {
        NdIndexOffsetHelper<int64_t, 3> index_helper(dy_vector.data());
        PoolKernelUtil<device_type, T, int64_t>::Maxpool2dBackwardCFirst(
            ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
      }
    } else if (data_format == "channels_last") {
      DimVector dy_vector;
      dy->shape_view().ToDimVector(&dy_vector);
      if (elem_num < GetMaxVal<int32_t>()) {
        NdIndexOffsetHelper<int32_t, 4> index_helper(dy_vector.data());
        PoolKernelUtil<device_type, T, int32_t>::Maxpool2dBackwardCLast(
            ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
      } else {
        NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());
        PoolKernelUtil<device_type, T, int64_t>::Maxpool2dBackwardCLast(
            ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
      }
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
    return CreatePoolOpKernelCache(ctx, 3);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);

    const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
    const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = y->shape_view().elem_cnt();
    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();

    DimVector y_vector(4);
    y_vector.at(0) = y->shape_view().At(0) * y->shape_view().At(1);
    y_vector.at(1) = y->shape_view().At(2);
    y_vector.at(2) = y->shape_view().At(3);
    y_vector.at(3) = y->shape_view().At(4);

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 4> index_helper(y_vector.data());
      PoolKernelUtil<device_type, T, int32_t>::Maxpool3dForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());
      PoolKernelUtil<device_type, T, int64_t>::Maxpool3dForward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    }
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
    return CreatePoolOpKernelCache(ctx, 3);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);

    const auto* pool_cache = dynamic_cast<const PoolOpKernelCache*>(cache);
    const MaxPoolParams3D& params_3d = pool_cache->GetParams3D();

    const int64_t elem_num = dy->shape_view().elem_cnt();
    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();

    DimVector dy_vector(4);
    dy_vector.at(0) = dy->shape_view().At(0) * dy->shape_view().At(1);
    dy_vector.at(1) = dy->shape_view().At(2);
    dy_vector.at(2) = dy->shape_view().At(3);
    dy_vector.at(3) = dy->shape_view().At(4);

    size_t out_bytes_size = dx->shape_view().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->stream(), dest, 0, out_bytes_size);

    if (elem_num < GetMaxVal<int32_t>()) {
      NdIndexOffsetHelper<int32_t, 4> index_helper(dy_vector.data());
      PoolKernelUtil<device_type, T, int32_t>::Maxpool3dBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    } else {
      NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());
      PoolKernelUtil<device_type, T, int64_t>::Maxpool3dBackward(
          ctx->stream(), index_helper, elem_num, src, dest, indice_ptr, params_3d);
    }
  };
};

#define REGISTER_POOL_KERNELS(device, dtype)                                            \
  REGISTER_USER_KERNEL("max_pool_1d")                                                   \
      .SetCreateFn<MaxPool1dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_1d_grad")                                              \
      .SetCreateFn<MaxPool1dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_2d")                                                   \
      .SetCreateFn<MaxPool2dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_2d_grad")                                              \
      .SetCreateFn<MaxPool2dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_3d")                                                   \
      .SetCreateFn<MaxPool3dKernel<device, dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("max_pool_3d_grad")                                              \
      .SetCreateFn<MaxPool3dGradKernel<device, dtype>>()                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                             \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

#define REGISTER_POOL_WITH_DEVICE(device) \
  REGISTER_POOL_KERNELS(device, int32_t)  \
  REGISTER_POOL_KERNELS(device, float)    \
  REGISTER_POOL_KERNELS(device, double)

REGISTER_POOL_WITH_DEVICE(DeviceType::kCPU)

#ifdef WITH_CUDA
REGISTER_POOL_WITH_DEVICE(DeviceType::kCUDA)
REGISTER_POOL_KERNELS(DeviceType::kCUDA, half)
#endif

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_POOL_KERNEL_UTIL, (DeviceType::kCPU),
                                 POOL_DATA_TYPE_CPU_SEQ, POOL_IDX_DATA_TYPE_SEQ);

}  // namespace oneflow
