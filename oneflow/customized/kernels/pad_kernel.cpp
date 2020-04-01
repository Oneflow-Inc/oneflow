#include "oneflow/core/framework/framework.h"
#include "oneflow/core/device/memory_copier.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

void SetShapeDimVector(const ShapeView& tensor_shape, const int64_t ndims,
                       const int64_t size_of_data_type, DimVector& shape_vec) {
  for (int64_t i = 0; i < ndims; ++i) { shape_vec[i] = tensor_shape.At(i); }
  shape_vec[ndims - 1] = shape_vec[ndims - 1] * size_of_data_type;
}

}  // namespace

template<DeviceType device_type, typename T>
class PadKernel final : public user_op::OpKernel {
 public:
  PadKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float constant_value = ctx->GetAttr<float>("constant_value");
    const auto paddings_vec = ctx->GetAttr<std::vector<int64_t>>("paddings");
    const int64_t ndims = x->shape().NumAxes();
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(x->data_type()));
    CHECK_EQ(paddings_vec.size(), ndims * 2);
    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), y->shape().elem_cnt(), constant_value,
                                     y->mut_dptr<T>());
    MemoryCopyNdDesc memory_copy_nd_desc;

    DimVector src_shape_vec(ndims);
    DimVector dst_shape_vec(ndims);
    SetShapeDimVector(x->shape(), ndims, size_of_data_type, src_shape_vec);
    SetShapeDimVector(y->shape(), ndims, size_of_data_type, dst_shape_vec);
    memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
    memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);

    DimVector src_pos_vec(ndims, 0);
    DimVector dst_pos_vec(ndims);
    for (int64_t i = 0; i < ndims; ++i) { dst_pos_vec[i] = paddings_vec[2 * i]; }
    dst_pos_vec[ndims - 1] *= size_of_data_type;

    memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
    memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
    memory_copy_nd_desc.extent = memory_copy_nd_desc.src_shape;
    MemoryCopyNdDesc reduced_memory_copy_nd_desc = memory_copy_nd_desc.CreateDimReducedDesc();
    std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
    device_memory_copier->Copy(ctx->device_ctx(), y->mut_dptr<T>(), x->dptr<T>(),
                               reduced_memory_copy_nd_desc);
  };
};

#define REGISTER_PAD_KERNEL(dev, dtype)                                                            \
  REGISTER_USER_KERNEL("pad")                                                                      \
      .SetCreateFn([](user_op::KernelInitContext* ctx) { return new PadKernel<dev, dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                                 \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);                \
        return ctx.device_type() == dev && y_desc->data_type() == GetDataType<dtype>::value;       \
      });

REGISTER_PAD_KERNEL(DeviceType::kGPU, double)
REGISTER_PAD_KERNEL(DeviceType::kCPU, double)
REGISTER_PAD_KERNEL(DeviceType::kGPU, float)
REGISTER_PAD_KERNEL(DeviceType::kCPU, float)

template<DeviceType device_type, typename T>
class PadGradKernel final : public user_op::OpKernel {
 public:
  PadGradKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}
  PadGradKernel() = default;
  ~PadGradKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto paddings_vec = ctx->GetAttr<std::vector<int64_t>>("paddings");
    const int64_t ndims = dy->shape().NumAxes();
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(dy->data_type()));

    MemoryCopyNdDesc memory_copy_nd_desc;

    DimVector src_shape_vec(ndims);
    DimVector dst_shape_vec(ndims);
    SetShapeDimVector(dy->shape(), ndims, size_of_data_type, src_shape_vec);
    SetShapeDimVector(dx->shape(), ndims, size_of_data_type, dst_shape_vec);
    memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
    memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);

    DimVector dst_pos_vec(ndims, 0);
    DimVector src_pos_vec(ndims);
    for (int64_t i = 0; i < ndims; ++i) { src_pos_vec[i] = paddings_vec[2 * i]; }
    src_pos_vec[ndims - 1] *= size_of_data_type;

    memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
    memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
    memory_copy_nd_desc.extent = memory_copy_nd_desc.dst_shape;
    MemoryCopyNdDesc reduced_memory_copy_nd_desc = memory_copy_nd_desc.CreateDimReducedDesc();
    std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
    device_memory_copier->Copy(ctx->device_ctx(), dx->mut_dptr<T>(), dy->dptr<T>(),
                               reduced_memory_copy_nd_desc);
  };
};

#define REGISTER_PAD_GRAD_KERNEL(dev, dtype)                                                  \
  REGISTER_USER_KERNEL("pad_grad")                                                            \
      .SetCreateFn(                                                                           \
          [](user_op::KernelInitContext* ctx) { return new PadGradKernel<dev, dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                            \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);         \
        return ctx.device_type() == dev && dx_desc->data_type() == GetDataType<dtype>::value; \
      });

REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, double)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, double)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_PAD_GRAD_KERNEL(DeviceType::kCPU, float)

}  // namespace oneflow
