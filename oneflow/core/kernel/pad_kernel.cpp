#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/memory_copier.h"
namespace oneflow {
namespace {

void SetShapeDimVector(const ShapeView& blob_shape, const int64_t ndims,
                       const int64_t size_of_data_type, DimVector& shape_vec) {
  const int64_t offset = blob_shape.NumAxes() - ndims;
  for (int64_t i = 0; i < ndims; ++i) { shape_vec[i] = blob_shape.At(i + offset); }
  if (offset > 0) {
    for (int64_t j = 0; j < offset; ++j) { shape_vec[0] = shape_vec[0] * blob_shape.At(j); }
  }
  shape_vec[ndims - 1] = shape_vec[ndims - 1] * size_of_data_type;
}

}  // namespace

template<DeviceType device_type, typename T>
class PadKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PadKernel);
  PadKernel() = default;
  ~PadKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* out_blob = BnInOp2Blob("out");
    const int32_t constant_value = this->op_conf().pad_conf().constant_value();
    Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), constant_value,
                        out_blob->AlignedByteSizeOfBlobBody());
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(in_blob->data_type()));
    MemoryCopyNdDesc memory_copy_nd_desc;
    const int32_t ndims = this->op_conf().pad_conf().paddings_size() / 2;

    DimVector src_shape_vec(ndims);
    DimVector dst_shape_vec(ndims);
    SetShapeDimVector(in_blob->shape(), ndims, size_of_data_type, src_shape_vec);
    SetShapeDimVector(out_blob->shape(), ndims, size_of_data_type, dst_shape_vec);
    memory_copy_nd_desc.src_shape = Shape(src_shape_vec);
    memory_copy_nd_desc.dst_shape = Shape(dst_shape_vec);

    DimVector dst_pos_vec(ndims);
    for (int64_t i = 0; i < ndims; ++i) {
      dst_pos_vec[i] = this->op_conf().pad_conf().paddings()[2 * i];
    }
    dst_pos_vec[ndims - 1] = dst_pos_vec[ndims - 1] * size_of_data_type;
    DimVector src_pos_vec(ndims, 0);

    memory_copy_nd_desc.dst_pos = NdIndex(dst_pos_vec);
    memory_copy_nd_desc.src_pos = NdIndex(src_pos_vec);
    memory_copy_nd_desc.extent = memory_copy_nd_desc.src_shape;
    std::unique_ptr<MemoryCopier> device_memory_copier(NewDefaultMemoryCopier(device_type));
    device_memory_copier->Copy(ctx.device_ctx, out_blob->mut_dptr(), in_blob->dptr(),
                               memory_copy_nd_desc);
  }
};

#define REGISTER_PAD_KERNEL(dev, dtype) \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kPadConf, dev, dtype, PadKernel<dev, dtype>)
REGISTER_PAD_KERNEL(DeviceType::kGPU, float);
REGISTER_PAD_KERNEL(DeviceType::kGPU, double);

}  // namespace oneflow
