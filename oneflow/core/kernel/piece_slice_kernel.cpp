#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PieceSliceKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PieceSliceKernel);
  PieceSliceKernel() = default;
  ~PieceSliceKernel() = default;

 private:
  void ForwardDenseShape(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    CHECK_EQ(in_blob->blob_desc().num_of_lod_levels(), 2);
    const int32_t out_size = this->op_conf().piece_slice_conf().out_size();
    CHECK_EQ(in_blob->length_lod_view().GetLength(0, 0), out_size);
    FOR_RANGE(size_t, i, 0, out_size) {
      auto* dense_shape_mut_view = BnInOp2Blob("out_" + std::to_string(i))->dense_shape_mut_view();
      dense_shape_mut_view->set_shape(in_blob->shape());
      dense_shape_mut_view->Set(0, in_blob->length_lod_view().GetLength(1, i));
    }
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    const int32_t instance_byte_size = in_blob->shape().Count(1) * sizeof(T);
    FOR_RANGE(size_t, i, 0, in_blob->length_lod_view().GetLength(0, 0)) {
      const char* src =
          in_blob->dptr<char>() + in_blob->offset_lod_view().GetOffset(1, i) * instance_byte_size;
      char* dst = BnInOp2Blob("out_" + std::to_string(i))->mut_dptr<char>();
      Memcpy<device_type>(ctx.device_ctx, dst, src,
                          in_blob->length_lod_view().GetLength(1, i) * instance_byte_size);
    }
  }
};

#define REGISTER_PIECE_SLICE_KERNEL(dtype)                                                      \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kPieceSliceConf, DeviceType::kCPU, dtype, \
                                        PieceSliceKernel<DeviceType::kCPU, dtype>)              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kPieceSliceConf, DeviceType::kGPU, dtype, \
                                        PieceSliceKernel<DeviceType::kGPU, dtype>)

REGISTER_PIECE_SLICE_KERNEL(float);
REGISTER_PIECE_SLICE_KERNEL(double);
REGISTER_PIECE_SLICE_KERNEL(int8_t);
REGISTER_PIECE_SLICE_KERNEL(int32_t);
REGISTER_PIECE_SLICE_KERNEL(int64_t);

}  // namespace oneflow
