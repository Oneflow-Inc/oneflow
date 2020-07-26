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
#include "oneflow/customized/ops/nn_util.h"

namespace oneflow {

namespace {

void GetDimVectorInBytes(const ShapeView& tensor_shape, const int64_t size_of_data_type,
                         DimVector& shape_vec) {
  int64_t num_axes = tensor_shape.NumAxes();
  for (int64_t i = 0; i < num_axes; ++i) { shape_vec[i] = tensor_shape.At(i); }
  shape_vec[num_axes - 1] = shape_vec[num_axes - 1] * size_of_data_type;
}

}  // namespace

template<DeviceType device_type, typename T>
class SamePaddingKernel final : public user_op::OpKernel {
 public:
  SamePaddingKernel() = default;
  ~SamePaddingKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t num_axes = x->shape().NumAxes();
    const std::string& padding = ctx->Attr<std::string>("padding");
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t> dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    std::vector<int64_t> padding_before(num_axes, 0);
    const size_t idx_offset = IdxOffset(data_format);
    const int32_t num_spatial_dims = x->shape().NumAxes() - 2;
    for (int32_t i = 0; i < num_spatial_dims; ++i) {
      int32_t padding_small = 0;
      int32_t padding_large = 0;
      CalcSamePadding(x->shape().At(idx_offset + i), kernel_size.at(i), dilation_rate.at(i),
                      strides.at(i), padding, &padding_small, &padding_large);
      if (padding == "same_lower") {
        padding_before[idx_offset + i] = padding_large;
      } else if (padding == "same_upper") {
        padding_before[idx_offset + i] = padding_small;
      } else {
        LOG(FATAL) << "padding must be same_lower or same_upper but get " << padding;
      }
      CHECK_EQ(y->shape().At(idx_offset + i),
               x->shape().At(idx_offset + i) + padding_small + padding_large);
    }
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(x->data_type()));
    CHECK_EQ(padding_before.size(), num_axes);
    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), y->shape().elem_cnt(), 0, y->mut_dptr<T>());
    MemoryCopyNdDesc memory_copy_nd_desc;

    DimVector src_shape_vec(num_axes);
    DimVector dst_shape_vec(num_axes);
    GetDimVectorInBytes(x->shape(), size_of_data_type, src_shape_vec);
    GetDimVectorInBytes(y->shape(), size_of_data_type, dst_shape_vec);
    memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
    memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);

    DimVector src_pos_vec(num_axes, 0);
    DimVector dst_pos_vec(padding_before.cbegin(), padding_before.cend());
    dst_pos_vec[num_axes - 1] *= size_of_data_type;

    memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
    memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
    memory_copy_nd_desc.extent = memory_copy_nd_desc.src_shape;
    MemoryCopyNdDesc reduced_memory_copy_nd_desc = memory_copy_nd_desc.CreateDimReducedDesc();

    std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
    device_memory_copier->Copy(ctx->device_ctx(), y->mut_dptr<T>(), x->dptr<T>(),
                               reduced_memory_copy_nd_desc);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SAME_PADDING_KERNEL(dev, dtype)         \
  REGISTER_USER_KERNEL("same_padding")                   \
      .SetCreateFn<SamePaddingKernel<dev, dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceType() == dev) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_SAME_PADDING_KERNEL(DeviceType::kGPU, double)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kGPU, float)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kGPU, int8_t)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kCPU, double)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kCPU, float)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_SAME_PADDING_KERNEL(DeviceType::kCPU, int8_t)

template<DeviceType device_type, typename T>
class SamePaddingGradKernel final : public user_op::OpKernel {
 public:
  SamePaddingGradKernel() = default;
  ~SamePaddingGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t num_axes = dy->shape().NumAxes();
    const std::string& padding = ctx->Attr<std::string>("padding");
    const std::string& data_format = ctx->Attr<std::string>("data_format");
    const std::vector<int32_t> kernel_size = ctx->Attr<std::vector<int32_t>>("kernel_size");
    const std::vector<int32_t> strides = ctx->Attr<std::vector<int32_t>>("strides");
    const std::vector<int32_t> dilation_rate = ctx->Attr<std::vector<int32_t>>("dilation_rate");
    std::vector<int64_t> padding_before(num_axes, 0);
    const size_t idx_offset = IdxOffset(data_format);
    const int32_t num_spatial_dims = dy->shape().NumAxes() - 2;
    for (int32_t i = 0; i < num_spatial_dims; ++i) {
      int32_t padding_small = 0;
      int32_t padding_large = 0;
      CalcSamePadding(dx->shape().At(idx_offset + i), kernel_size.at(i), dilation_rate.at(i),
                      strides.at(i), padding, &padding_small, &padding_large);
      if (padding == "SAME_LOWER") {
        padding_before[idx_offset + i] = padding_large;
      } else {  //"SAME_UPPER"
        padding_before[idx_offset + i] = padding_small;
      }
      CHECK_EQ(dy->shape().At(idx_offset + i),
               dx->shape().At(idx_offset + i) + padding_small + padding_large);
    }

    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(dy->data_type()));

    MemoryCopyNdDesc memory_copy_nd_desc;

    DimVector src_shape_vec(num_axes);
    DimVector dst_shape_vec(num_axes);
    GetDimVectorInBytes(dy->shape(), size_of_data_type, src_shape_vec);
    GetDimVectorInBytes(dx->shape(), size_of_data_type, dst_shape_vec);
    memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
    memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);

    DimVector dst_pos_vec(num_axes, 0);
    DimVector src_pos_vec(padding_before.cbegin(), padding_before.cend());
    src_pos_vec[num_axes - 1] *= size_of_data_type;

    memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
    memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
    memory_copy_nd_desc.extent = memory_copy_nd_desc.dst_shape;
    MemoryCopyNdDesc reduced_memory_copy_nd_desc = memory_copy_nd_desc.CreateDimReducedDesc();

    std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
    device_memory_copier->Copy(ctx->device_ctx(), dx->mut_dptr<T>(), dy->dptr<T>(),
                               reduced_memory_copy_nd_desc);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SAME_PADDING_GRAD_KERNEL(dev, dtype)    \
  REGISTER_USER_KERNEL("same_padding_grad")              \
      .SetCreateFn<SamePaddingGradKernel<dev, dtype>>()  \
      .SetIsMatchedHob((user_op::HobDeviceType() == dev) \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kGPU, double)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kGPU, int8_t)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kCPU, double)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_SAME_PADDING_GRAD_KERNEL(DeviceType::kCPU, int8_t)

}  // namespace oneflow
