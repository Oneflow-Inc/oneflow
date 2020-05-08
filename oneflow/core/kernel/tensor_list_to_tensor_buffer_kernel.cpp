#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/common/tensor_buffer.h"

namespace oneflow {

class TensorListToTensorBufferKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorListToTensorBufferKernel);
  TensorListToTensorBufferKernel() = default;
  ~TensorListToTensorBufferKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

void TensorListToTensorBufferKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  TensorView in_tensor_view = in_blob->BeginTensor();
  TensorBuffer* out_buffer = out_blob->mut_dptr<TensorBuffer>();
  while (!in_blob->IsEndTensor(in_tensor_view)) {
    CHECK_EQ(in_tensor_view.shape().At(0), 1);
    DimVector dim_vec(in_tensor_view.shape().NumAxes() - 1);
    FOR_RANGE(int, i, 0, dim_vec.size()) { dim_vec.at(i) = in_tensor_view.shape().At(i + 1); }
    out_buffer->Resize(Shape(dim_vec), in_blob->data_type());
    memcpy(out_buffer->mut_data(), in_tensor_view.dptr(), out_buffer->nbytes());
    out_buffer += 1;
    in_blob->MoveToNextTensor(&in_tensor_view);
  }
}

NEW_REGISTER_KERNEL(OperatorConf::kTensorListToTensorBufferConf, TensorListToTensorBufferKernel)
    .SetIsMatchedPred([](const KernelConf& conf) {
      return (conf.op_attribute().op_conf().device_type() == DeviceType::kCPU)
             && (conf.data_type() == DataType::kTensorBuffer);
    });

}  // namespace oneflow
