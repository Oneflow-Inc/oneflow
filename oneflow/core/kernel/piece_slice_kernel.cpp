#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class PieceSliceKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PieceSliceKernel);
  PieceSliceKernel() = default;
  ~PieceSliceKernel() = default;

 private:
  void ForwardShape(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    // do nothing
  }
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    CHECK_EQ(in_blob->blob_desc().is_tensor_list(), true);
    const int32_t out_size = this->op_conf().piece_slice_conf().out_size();
    CHECK_EQ(in_blob->total_num_of_tensors(), out_size);
    auto in_tensor = in_blob->BeginTensor();
    FOR_RANGE(size_t, i, 0, out_size) {
      Blob* out_blob = BnInOp2Blob("out_" + std::to_string(i));
      auto tensor_inserter = out_blob->tensor_back_inserter();
      tensor_inserter.ReserveOneEmptyTensorList();
      auto* out_tensor = tensor_inserter.add_tensor();
      DimVector dim_vec;
      in_tensor.shape().ToDimVector(&dim_vec);
      dim_vec.erase(dim_vec.begin());
      out_tensor->set_shape(Shape(dim_vec));
      Memcpy<device_type>(ctx.device_ctx, out_tensor->mut_dptr(), in_tensor.dptr(),
                          in_tensor.ByteSize());
      in_blob->MoveToNextTensor(&in_tensor);
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
