#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/memory_copier.h"
namespace oneflow {

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
    Memset<device_type>(ctx.device_ctx, out_blob->mut_dptr<T>(), 0,
                        out_blob->AlignedByteSizeOfBlobBody());
    const int64_t size_of_data_type = static_cast<int64_t>(GetSizeOfDataType(in_blob->data_type()));
    MemoryCopyNdDesc memory_copy_nd_desc;

    int64_t ndims = in_blob->shape().NumAxes();
    memory_copy_nd_desc.dst_shape = out_blob->static_shape();
    memory_copy_nd_desc.dst_shape.Set(ndims-1, out_blob->shape().At(ndims-1) * size_of_data_type);
    memory_copy_nd_desc.src_shape = in_blob->static_shape();
    memory_copy_nd_desc.src_shape.Set(ndims-1, in_blob->shape().At(ndims-1) * size_of_data_type);
    
    DimVector dst_pos_vec(in_blob->shape().NumAxes()) ;
    for (int64_t i = 0; i < ndims; ++i) {
        dst_pos_vec[i] = this->op_conf().pad_conf().paddings()[2*i];
    }
    DimVector src_pos_vec(in_blob->shape().NumAxes(), 0);

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
