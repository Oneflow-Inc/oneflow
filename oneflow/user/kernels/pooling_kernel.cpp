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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/pooling_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

// Fill ShapeView into dim vector
DimVector ShapeViewToDimVector(const ShapeView& tensor_shape) {
  int32_t ndims = tensor_shape.NumAxes();
  DimVector shape_vec(ndims);
  for (int32_t i = 0; i < ndims; ++i) { shape_vec[i] = tensor_shape.At(i); }
  shape_vec[ndims - 1] = shape_vec[ndims - 1];
  return shape_vec;
}

}  // namespace

template<typename T>
struct PoolingKernelUtil<DeviceType::kCPU, T> {
  static void Maxpool2dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4> index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const std::vector<int32_t> padding_before, const int64_t n_batch,
                               const int64_t n_channel, const int64_t x_height,
                               const int64_t x_width, const int64_t y_height, const int64_t y_width,
                               const std::vector<int32_t> kernel_size,
                               const std::vector<int32_t> stride,
                               const std::vector<int32_t> dilation, const bool return_indices,
                               const bool ceil_mode) {
    T maxval = -std::numeric_limits<T>::infinity();
    Maxpool2dFarwardCompute<T>(index_helper, elem_num, maxval, src, dest, indice_ptr, padding_before[0],
                      padding_before[1], n_batch, n_channel, x_height, x_width, y_height, y_width,
                      kernel_size[0], kernel_size[1], stride[0], stride[1], dilation[0],
                      dilation[1], return_indices, ceil_mode);
  }

  static void Maxpool2dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 4> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const int64_t n_batch,
                                const int64_t n_channel, const int64_t src_height,
                                const int64_t src_width, const int64_t dst_height,
                                const int64_t dst_width, const bool return_indices,
                                const bool ceil_mode) {
    Maxpool2dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
                       src_height, src_width, dst_height, dst_width, return_indices, ceil_mode);
  }


  static void Maxpool3dForward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper,
                               const int64_t elem_num, const T* src, T* dest, int64_t* indice_ptr,
                               const std::vector<int32_t> padding_before, const int64_t n_batch,
                               const int64_t n_channel, const int64_t x_time, const int64_t x_height,
                               const int64_t x_width, const int64_t y_time, const int64_t y_height, const int64_t y_width,
                               const std::vector<int32_t> kernel_size, const std::vector<int32_t> stride,
                               const std::vector<int32_t> dilation, const bool return_indices,
                               const bool ceil_mode) {
    T maxval = -std::numeric_limits<T>::infinity();
    Maxpool3dFarwardCompute<T>(index_helper, elem_num, maxval, src, dest, indice_ptr, padding_before[0], padding_before[1],
                      padding_before[2], n_batch, n_channel, x_time, x_height, x_width, y_time, y_height, y_width,
                      kernel_size[0], kernel_size[1], kernel_size[2], stride[0], stride[1], stride[2], 
                      dilation[0], dilation[1], dilation[2], return_indices, ceil_mode);
  }


  static void Maxpool3dBackward(DeviceCtx* ctx, const NdIndexOffsetHelper<int64_t, 5> index_helper,
                                const int64_t elem_num, const T* src, T* dest,
                                const int64_t* indice_ptr, const int64_t n_batch, const int64_t n_channel, 
                                const int64_t src_time, const int64_t src_height, const int64_t src_width, 
                                const int64_t dst_time, const int64_t dst_height, const int64_t dst_width, 
                                const bool return_indices, const bool ceil_mode
                                ) {
    Maxpool3dBackwardCompute<T>(index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
                       src_time, src_height, src_width, dst_time, dst_height, dst_width, return_indices, ceil_mode);
  }

};

template<DeviceType device_type, typename T>
class MaxPool2dKernel final : public user_op::OpKernel {
 public:
  MaxPool2dKernel() = default;
  ~MaxPool2dKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    printf("Enter MaxPool2dKernel >>>>>>>>>>>>>>>>>>>>>>> Compute()\n");
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    const std::string data_format = ctx->Attr<std::string>("data_format");
    const std::string padding = ctx->Attr<std::string>("padding");
    const std::vector<int32_t> padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const std::vector<int32_t> padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation");
    const bool return_indices = ctx->Attr<bool>("return_indices");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    int32_t c_idx, h_idx, w_idx;
    if (data_format == "channels_first") {
      c_idx = 1;
      h_idx = 2;
      w_idx = 3;
    } else if (data_format == "channels_last") {
      c_idx = 3;
      h_idx = 1;
      w_idx = 2;
    } else {
      UNIMPLEMENTED();
    }
    const int64_t n_batch = x->shape().At(0);
    const int64_t n_channel = x->shape().At(c_idx);
    const int64_t x_height = x->shape().At(h_idx);
    const int64_t x_width = x->shape().At(w_idx);
    const int64_t y_height = y->shape().At(h_idx);
    const int64_t y_width = y->shape().At(w_idx);
    const int64_t elem_num = y->shape().elem_cnt();
    printf("elem_num:>>>>>>>>>>>>>>>>>>>>>>> %ld \n", elem_num);

    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();
    DimVector y_vector = ShapeViewToDimVector(y->shape());
    NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());

    PoolingKernelUtil<device_type, T>::Maxpool2dForward(
        ctx->device_ctx(), index_helper, elem_num, src, dest, indice_ptr, padding_before, n_batch,
        n_channel, x_height, x_width, y_height, y_width, kernel_size, stride, dilation,
        return_indices, ceil_mode);
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
    const std::string data_format = ctx->Attr<std::string>("data_format");
    const std::string padding = ctx->Attr<std::string>("padding");
    const std::vector<int32_t> padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const std::vector<int32_t> padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation");
    const bool return_indices = ctx->Attr<bool>("return_indices");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    int32_t c_idx, h_idx, w_idx;
    if (data_format == "channels_first") {
      c_idx = 1;
      h_idx = 2;
      w_idx = 3;
    } else if (data_format == "channels_last") {
      c_idx = 3;
      h_idx = 1;
      w_idx = 2;
    } else {
      UNIMPLEMENTED();
    }
    const int64_t n_batch = dy->shape().At(0);
    const int64_t n_channel = dy->shape().At(c_idx);
    const int64_t src_height = dy->shape().At(h_idx);
    const int64_t src_width = dy->shape().At(w_idx);
    const int64_t dst_height = dx->shape().At(h_idx);
    const int64_t dst_width = dx->shape().At(w_idx);
    const int64_t elem_num = dy->shape().elem_cnt();

    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector = ShapeViewToDimVector(dy->shape());
    NdIndexOffsetHelper<int64_t, 4> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);

    PoolingKernelUtil<device_type, T>::Maxpool2dBackward(
        ctx->device_ctx(), index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
        src_height, src_width, dst_height, dst_width, return_indices, ceil_mode);
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
    const std::string data_format = ctx->Attr<std::string>("data_format");
    const std::string padding = ctx->Attr<std::string>("padding");
    const std::vector<int32_t> padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const std::vector<int32_t> padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation");
    const bool return_indices = ctx->Attr<bool>("return_indices");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    int32_t c_idx, t_idx, h_idx, w_idx;
    if (data_format == "channels_first") {
      c_idx = 1;
      t_idx = 2;
      h_idx = 3;
      w_idx = 4;
    } else if (data_format == "channels_last") {
      c_idx = 4;
      t_idx = 1;
      h_idx = 2;
      w_idx = 3;
    } else {
      UNIMPLEMENTED();
    }
    const int64_t n_batch = x->shape().At(0);
    const int64_t n_channel = x->shape().At(c_idx);
    const int64_t x_time = x->shape().At(t_idx);
    const int64_t x_height = x->shape().At(h_idx);
    const int64_t x_width = x->shape().At(w_idx);
    const int64_t y_time = y->shape().At(t_idx);
    const int64_t y_height = y->shape().At(h_idx);
    const int64_t y_width = y->shape().At(w_idx);
    const int64_t elem_num = y->shape().elem_cnt();

    const T* src = x->dptr<T>();
    T* dest = y->mut_dptr<T>();
    int64_t* indice_ptr = indice->mut_dptr<int64_t>();
    DimVector y_vector = ShapeViewToDimVector(y->shape());
    NdIndexOffsetHelper<int64_t, 5> index_helper(y_vector.data());

    PoolingKernelUtil<device_type, T>::Maxpool3dForward(
        ctx->device_ctx(), index_helper, elem_num, src, dest, indice_ptr, padding_before, n_batch,
        n_channel, x_time, x_height, x_width, y_time, y_height, y_width, kernel_size, stride, dilation,
        return_indices, ceil_mode);
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
    printf("Enter MaxPool3dGradKernel >>>>>>>>>>>>>>>>>>>>>>> Compute()\n");
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* indice = ctx->Tensor4ArgNameAndIndex("indice", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const std::string data_format = ctx->Attr<std::string>("data_format");
    const std::string padding = ctx->Attr<std::string>("padding");
    const std::vector<int32_t> padding_before = ctx->Attr<std::vector<int32_t>>("padding_before");
    const std::vector<int32_t> padding_after = ctx->Attr<std::vector<int32_t>>("padding_after");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> stride = ctx->Attr<std::vector<int32_t>>("stride");
    const std::vector<int32_t> dilation = ctx->Attr<std::vector<int32_t>>("dilation");
    const bool return_indices = ctx->Attr<bool>("return_indices");
    const bool ceil_mode = ctx->Attr<bool>("ceil_mode");

    int32_t c_idx, t_idx, h_idx, w_idx;
    if (data_format == "channels_first") {
      c_idx = 1;
      t_idx = 2;
      h_idx = 3;
      w_idx = 4;
    } else if (data_format == "channels_last") {
      c_idx = 4;
      t_idx = 1;
      h_idx = 2;
      w_idx = 3;
    } else {
      UNIMPLEMENTED();
    }
    const int64_t n_batch = dy->shape().At(0);
    const int64_t n_channel = dy->shape().At(c_idx);
    const int64_t src_time = dy->shape().At(t_idx);
    const int64_t src_height = dy->shape().At(h_idx);
    const int64_t src_width = dy->shape().At(w_idx);
    const int64_t dst_time = dx->shape().At(t_idx);
    const int64_t dst_height = dx->shape().At(h_idx);
    const int64_t dst_width = dx->shape().At(w_idx);
    const int64_t elem_num = dy->shape().elem_cnt();
    printf("elem_num:>>>>>>>>>>>>>>>>>>>>>>> %ld \n", elem_num);

    const T* src = dy->dptr<T>();
    const int64_t* indice_ptr = indice->dptr<int64_t>();
    T* dest = dx->mut_dptr<T>();
    DimVector dy_vector = ShapeViewToDimVector(dy->shape());
    NdIndexOffsetHelper<int64_t, 5> index_helper(dy_vector.data());

    size_t out_bytes_size = dx->shape().elem_cnt() * GetSizeOfDataType(dx->data_type());
    Memset<device_type>(ctx->device_ctx(), dest, 0, out_bytes_size);

    PoolingKernelUtil<device_type, T>::Maxpool3dBackward(
        ctx->device_ctx(), index_helper, elem_num, src, dest, indice_ptr, n_batch, n_channel,
        src_time, src_height, src_width, dst_time, dst_height, dst_width, return_indices, ceil_mode);
  };
};


#define REGISTER_POOLING_KERNELS(device, dtype)                                        \
  REGISTER_USER_KERNEL("maxpool_2d")                                                   \
      .SetCreateFn<MaxPool2dKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_2d_grad")                                              \
      .SetCreateFn<MaxPool2dGradKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_3d")                                                   \
      .SetCreateFn<MaxPool3dKernel<device, dtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("maxpool_3d_grad")                                              \
      .SetCreateFn<MaxPool3dGradKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                             \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

#define REGISTER_POOLING_WITH_DEVICE(device) \
  REGISTER_POOLING_KERNELS(device, float)    \
  REGISTER_POOLING_KERNELS(device, double)   \
  REGISTER_POOLING_KERNELS(device, int32_t)  \
  REGISTER_POOLING_KERNELS(device, int64_t)

REGISTER_POOLING_WITH_DEVICE(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_POOLING_WITH_DEVICE(DeviceType::kGPU)
// REGISTER_POOLING_KERNELS(DeviceType::kGPU, float16)
#endif

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL_UTIL, (DeviceType::kCPU),
                                 POOLING_DATA_TYPE_CPU_SEQ);

}  // namespace oneflow
