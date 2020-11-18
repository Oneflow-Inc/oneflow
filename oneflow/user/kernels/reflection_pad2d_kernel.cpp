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
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

// Get dim vector with size of data type
void GetDimVectorInBytes(const ShapeView& tensor_shape, const int64_t size_of_data_type,
                         DimVector& shape_vec) {
  int64_t ndims = tensor_shape.NumAxes();
  for (int64_t i = 0; i < ndims; ++i) { shape_vec[i] = tensor_shape.At(i); }
  // Replace the last dimension of the dim vector with: the number of elements n × the number of
  // bytes of the data type
  shape_vec[ndims - 1] = shape_vec[ndims - 1] * size_of_data_type;
}

// Fill ShapeView into dim vector
DimVector ShapeViewToDimVector(const ShapeView& tensor_shape) {
  int64_t ndims = tensor_shape.NumAxes();
  DimVector shape_vec(ndims);
  for (int64_t i = 0; i < ndims; ++i) { shape_vec[i] = tensor_shape.At(i); }
  shape_vec[ndims - 1] = shape_vec[ndims - 1];
  return shape_vec;
}

// Fill input elements(x) to output body(y)
template<DeviceType device_type, typename T>
void FillBodyElements(const int64_t ndims, const user_op::Tensor* x, user_op::Tensor* y,
                      const int64_t sizeof_dtype, user_op::KernelComputeContext* ctx,
                      bool is_backward) {
  const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
  MemoryCopyNdDesc memory_copy_nd_desc;
  DimVector src_shape_vec(ndims);
  DimVector dst_shape_vec(ndims);

  GetDimVectorInBytes(x->shape(), sizeof_dtype, src_shape_vec);
  GetDimVectorInBytes(y->shape(), sizeof_dtype, dst_shape_vec);
  memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
  memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);
  if (is_backward == true) {
    DimVector dst_pos_vec(ndims, 0);
    DimVector src_pos_vec(padding.cbegin(), padding.cend());
    src_pos_vec[ndims - 1] *= sizeof_dtype;
    memory_copy_nd_desc.extent = memory_copy_nd_desc.dst_shape;
    memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
    memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
  } else {
    DimVector src_pos_vec(ndims, 0);
    DimVector dst_pos_vec(padding.cbegin(), padding.cend());
    dst_pos_vec[ndims - 1] *= sizeof_dtype;
    memory_copy_nd_desc.extent = memory_copy_nd_desc.src_shape;
    memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
    memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
  }
  MemoryCopyNdDesc reduced_memory_copy_nd_desc = memory_copy_nd_desc.CreateDimReducedDesc();
  std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
  device_memory_copier->Copy(ctx->device_ctx(), y->mut_dptr<T>(), x->dptr<T>(),
                             reduced_memory_copy_nd_desc);
}

// Padding the diagonal elements around index_vector and assign values
template<DeviceType device_type, typename T>
void PaddingDiagonalElements(const int64_t w_idx, NdIndexOffsetHelper<int64_t, 4>& index_helper,
                             const std::vector<int64_t>& index_vector, const user_op::Tensor* x,
                             user_op::Tensor* y, user_op::KernelComputeContext* ctx) {
  const int64_t sizeof_dtype = static_cast<int64_t>(GetSizeOfDataType(x->data_type()));
  const std::string data_format = ctx->Attr<std::string>("data_format");
  const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
  int64_t padding_w;
  padding_w = padding[w_idx];

  FOR_RANGE(int, i, 0, index_vector.size()) {
    int64_t coord_y[4];
    index_helper.OffsetToNdIndex(index_vector[i], coord_y);
    int64_t dest_w;
    int64_t dest_index;
    int64_t dest_coords[4];
    if (coord_y[w_idx] < padding_w) {
      // left part
      dest_w = 2 * padding_w - coord_y[w_idx];
    } else {
      // right part
      dest_w = 2 * (padding_w + x->shape().At(w_idx) - 1) - coord_y[w_idx];
    }
    dest_coords[0] = coord_y[0];
    dest_coords[1] = coord_y[1];
    if (data_format == "NCHW") {
      dest_coords[2] = coord_y[2];
      dest_coords[3] = dest_w;
    } else {
      dest_coords[2] = dest_w;
      dest_coords[3] = coord_y[3];
    }
    dest_index = index_helper.NdIndexToOffset(dest_coords);
    Memcpy<device_type>(ctx->device_ctx(), y->mut_dptr<T>() + index_vector[i],
                        y->mut_dptr<T>() + dest_index, sizeof_dtype);
  }
}

}  // namespace

template<DeviceType device_type, typename T>
class ReflectionPad2dKernel final : public user_op::OpKernel {
 public:
  ReflectionPad2dKernel() = default;
  ~ReflectionPad2dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
    const std::string data_format = ctx->Attr<std::string>("data_format");
    const int64_t ndims = x->shape().NumAxes();
    const int64_t sizeof_dtype = static_cast<int64_t>(GetSizeOfDataType(x->data_type()));
    CHECK_EQ(padding.size(), ndims);

    FillBodyElements<device_type, T>(ndims, x, y, sizeof_dtype, ctx, false);

    int64_t padding_h, padding_w, h_idx, w_idx;
    if (data_format == "NCHW") {
      h_idx = 2;
      w_idx = 3;
    } else {
      h_idx = 1;
      w_idx = 2;
    }
    padding_h = padding[h_idx];
    padding_w = padding[w_idx];
    const ShapeView& x_shape = x->shape();
    // elements index vector of diagonal elements
    std::vector<int64_t> index_vector;

    DimVector y_vector = ShapeViewToDimVector(y->shape());
    NdIndexOffsetHelper<int64_t, 4> index_helper(y_vector.data());

    FOR_RANGE(int, i, 0, y->shape().elem_cnt()) {
      // Traverse one-dimensional array y
      int64_t coord_y[ndims];
      index_helper.OffsetToNdIndex(i, coord_y);
      int64_t x_h = coord_y[h_idx] - padding_h;
      int64_t x_w = coord_y[w_idx] - padding_w;
      if (x_h < 0 || x_h >= x_shape.At(h_idx) || x_w < 0 || x_w >= x_shape.At(w_idx)) {
        // Indicates that the element is no longer in the original x range (the data to be padding
        // outside)
        int64_t dest_coords[4];
        int64_t dest_index;
        // Determine whether it is a diagonal element
        if ((x_h >= 0 && x_h < x_shape.At(2))) {
          // Within the left and right range lines, non-diagonal elements
          int64_t dest_w;
          if (x_w < 0) {
            // left part
            dest_w = 2 * padding_w - coord_y[w_idx];
          } else {
            // right pary
            dest_w = 2 * (padding_w + x_shape.At(w_idx) - 1) - coord_y[w_idx];
          }
          dest_coords[0] = coord_y[0];
          dest_coords[1] = coord_y[1];
          if (data_format == "NCHW") {
            dest_coords[2] = coord_y[2];
            dest_coords[3] = dest_w;
          } else {
            dest_coords[2] = dest_w;
            dest_coords[3] = coord_y[3];
          }
          dest_index = index_helper.NdIndexToOffset(dest_coords);
          Memcpy<device_type>(ctx->device_ctx(), y->mut_dptr<T>() + i,
                              y->mut_dptr<T>() + dest_index, sizeof_dtype);
        } else if (x_w >= 0 && x_w < x_shape.At(w_idx)) {
          // Within the upper and lower range lines, non-diagonal elements
          int64_t dest_h;
          if (x_h < 0) {
            // upper part
            dest_h = 2 * padding_h - coord_y[h_idx];
          } else {
            // lower part
            dest_h = 2 * (padding_h + x_shape.At(h_idx) - 1) - coord_y[h_idx];
          }
          dest_coords[0] = coord_y[0];
          dest_coords[3] = coord_y[3];
          if (data_format == "NCHW") {
            dest_coords[1] = coord_y[1];
            dest_coords[2] = dest_h;
          } else {
            dest_coords[1] = dest_h;
            dest_coords[2] = coord_y[2];
          }
          dest_index = index_helper.NdIndexToOffset(dest_coords);
          Memcpy<device_type>(ctx->device_ctx(), y->mut_dptr<T>() + i,
                              y->mut_dptr<T>() + dest_index, sizeof_dtype);
        } else {
          // Diagonal element
          index_vector.push_back(i);
        }
      }
    }

    PaddingDiagonalElements<device_type, T>(w_idx, index_helper, index_vector, x, y, ctx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REFLECTION_PAD2D_KERNELS(dev, dtype)   \
  REGISTER_USER_KERNEL("reflection_pad2d")              \
      .SetCreateFn<ReflectionPad2dKernel<dev, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, float)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, double)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, float16)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int32_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int64_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kGPU, int8_t)
#endif
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, float)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, double)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int32_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int64_t)
REGISTER_REFLECTION_PAD2D_KERNELS(DeviceType::kCPU, int8_t)

template<DeviceType device_type, typename T>
class ReflectionPad2dGradKernel final : public user_op::OpKernel {
 public:
  ReflectionPad2dGradKernel() = default;
  ~ReflectionPad2dGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto& padding = ctx->Attr<std::vector<int64_t>>("padding");
    const int64_t ndims = dy->shape().NumAxes();
    const int64_t sizeof_dtype = static_cast<int64_t>(GetSizeOfDataType(dy->data_type()));
    CHECK_EQ(padding.size(), ndims);

    FillBodyElements<device_type, T>(ndims, dy, dx, sizeof_dtype, ctx, true);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(dev, dtype)  \
  REGISTER_USER_KERNEL("reflection_pad2d_grad")             \
      .SetCreateFn<ReflectionPad2dGradKernel<dev, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev)     \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, float)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, double)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, float16)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int32_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int64_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kGPU, int8_t)
#endif
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, float)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, double)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int32_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int64_t)
REGISTER_REFLECTION_PAD2D_GRAD_KERNELS(DeviceType::kCPU, int8_t)

}  // namespace oneflow