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

namespace oneflow {

namespace {

void GetDimVectorInBytes(const ShapeView& tensor_shape, const int64_t size_of_data_type,
                         DimVector& shape_vec) {
  int64_t ndims = tensor_shape.NumAxes();
  for (int64_t i = 0; i < ndims; ++i) { shape_vec[i] = tensor_shape.At(i); }
  shape_vec[ndims - 1] = shape_vec[ndims - 1] * size_of_data_type;
}

template<typename T>
T GetDtypeMatchedValue(double floating, int64_t integral);

template<>
float16 GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<float16>(floating);
}

template<>
float GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<float>(floating);
}

template<>
double GetDtypeMatchedValue(double floating, int64_t integral) {
  return floating;
}

template<>
int8_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<int8_t>(integral);
}

template<>
int32_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return static_cast<int32_t>(integral);
}

template<>
int64_t GetDtypeMatchedValue(double floating, int64_t integral) {
  return integral;
}

}  // namespace

template<DeviceType device_type, typename T>
class PadKernel final : public user_op::OpKernel {
 public:
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T constant_value = GetDtypeMatchedValue<T>(ctx->Attr<double>("floating_constant_value"),
                                                     ctx->Attr<int64_t>("integral_constant_value"));
    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const int64_t ndims = x->shape().NumAxes();
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(x->data_type()));
    CHECK_EQ(padding_before.size(), ndims);
    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), y->shape().elem_cnt(),
                                     static_cast<T>(constant_value), y->mut_dptr<T>());
    MemoryCopyNdDesc memory_copy_nd_desc;

    DimVector src_shape_vec(ndims);
    DimVector dst_shape_vec(ndims);
    GetDimVectorInBytes(x->shape(), size_of_data_type, src_shape_vec);
    GetDimVectorInBytes(y->shape(), size_of_data_type, dst_shape_vec);
    memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
    memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);

    DimVector src_pos_vec(ndims, 0);
    DimVector dst_pos_vec(padding_before.cbegin(), padding_before.cend());
    dst_pos_vec[ndims - 1] *= size_of_data_type;

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

#define REGISTER_PAD_KERNEL(dev, dtype)                                             \
  REGISTER_USER_KERNEL("pad").SetCreateFn<PadKernel<dev, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == dev)                                              \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_PAD_KERNEL(DeviceType::kGPU, double)
REGISTER_PAD_KERNEL(DeviceType::kGPU, float)
REGISTER_PAD_KERNEL(DeviceType::kGPU, float16)
REGISTER_PAD_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_PAD_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_PAD_KERNEL(DeviceType::kGPU, int8_t)
#endif
REGISTER_PAD_KERNEL(DeviceType::kCPU, double)
REGISTER_PAD_KERNEL(DeviceType::kCPU, float)
REGISTER_PAD_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_PAD_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_PAD_KERNEL(DeviceType::kCPU, int8_t)

template<DeviceType device_type, typename T>
class PadGradKernel final : public user_op::OpKernel {
 public:
  PadGradKernel() = default;
  ~PadGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const int64_t ndims = dy->shape().NumAxes();
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(dy->data_type()));

    MemoryCopyNdDesc memory_copy_nd_desc;

    DimVector src_shape_vec(ndims);
    DimVector dst_shape_vec(ndims);
    GetDimVectorInBytes(dy->shape(), size_of_data_type, src_shape_vec);
    GetDimVectorInBytes(dx->shape(), size_of_data_type, dst_shape_vec);
    memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
    memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);

    DimVector dst_pos_vec(ndims, 0);
    DimVector src_pos_vec(padding_before.cbegin(), padding_before.cend());
    src_pos_vec[ndims - 1] *= size_of_data_type;

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

#define REGISTER_PAD_GRAD_KERNEL(dev, dtype)            \
  REGISTER_USER_KERNEL("pad_grad")                      \
      .SetCreateFn<PadGradKernel<dev, dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev) \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, double)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, float16)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, int8_t)
#endif
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, double)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, int8_t)

}  // namespace oneflow
