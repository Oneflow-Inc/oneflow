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
#include "oneflow/core/kernel/cuda_graph_support.h"
#if defined(WITH_CUDA) && CUDA_VERSION >= 11000
#include "oneflow/core/device/cuda_pseudo_bfloat16.h"
#endif

namespace oneflow {

namespace {

void GetDimVectorInBytes(const ShapeView& tensor_shape, const int64_t size_of_data_type,
                         DimVector& shape_vec) {
  int64_t ndims = tensor_shape.NumAxes();
  for (int64_t i = 0; i < ndims; ++i) { shape_vec[i] = tensor_shape.At(i); }
  shape_vec[ndims - 1] = shape_vec[ndims - 1] * size_of_data_type;
}

const void* GetDtypeMatchedValuePtr(const DataType data_type, double floating, int64_t integral) {
  if (data_type == kFloat) {
    static const float val = static_cast<float>(floating);
    return static_cast<const void*>(&val);
  } else if (data_type == kDouble) {
    static const double val = floating;
    return static_cast<const void*>(&val);
  } else if (data_type == kFloat16) {
    static const float16 val = static_cast<float16>(floating);
    return static_cast<const void*>(&val);
  } else if (data_type == kInt8) {
    static const int8_t val = static_cast<int8_t>(integral);
    return static_cast<const void*>(&val);
  } else if (data_type == kInt32) {
    static const int32_t val = static_cast<int32_t>(integral);
    return static_cast<const void*>(&val);
  } else if (data_type == kInt64) {
    static const int64_t val = static_cast<int64_t>(integral);
    return static_cast<const void*>(&val);
  } else if (data_type == kBFloat16) {
#if defined(WITH_CUDA) && CUDA_VERSION >= 11000
    static const nv_bfloat16 val = static_cast<nv_bfloat16>(floating);
    return static_cast<const void*>(&val);
#else
    UNIMPLEMENTED();
#endif
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

template<DeviceType device_type>
class PadKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
    const int64_t ndims = x->shape().NumAxes();
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(x->data_type()));
    CHECK_EQ(padding_before.size(), ndims);
    NewKernelUtil<device_type>::Fill(
        ctx->device_ctx(), y->shape().elem_cnt(), y->data_type(),
        GetDtypeMatchedValuePtr(y->data_type(), ctx->Attr<double>("floating_constant_value"),
                                ctx->Attr<int64_t>("integral_constant_value")),
        y->mut_dptr());
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
    device_memory_copier->Copy(ctx->device_ctx(), y->mut_dptr(), x->dptr(),
                               reduced_memory_copy_nd_desc);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PAD_KERNEL(dev)                                             \
  REGISTER_USER_KERNEL("pad").SetCreateFn<PadKernel<dev>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == dev));

#ifdef WITH_CUDA
REGISTER_PAD_KERNEL(DeviceType::kGPU)
#endif
REGISTER_PAD_KERNEL(DeviceType::kCPU)

template<DeviceType device_type>
class PadGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
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
    device_memory_copier->Copy(ctx->device_ctx(), dx->mut_dptr(), dy->dptr(),
                               reduced_memory_copy_nd_desc);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PAD_GRAD_KERNEL(dev)    \
  REGISTER_USER_KERNEL("pad_grad")       \
      .SetCreateFn<PadGradKernel<dev>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == dev));

#ifdef WITH_CUDA
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU)
#endif
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU)

}  // namespace oneflow
