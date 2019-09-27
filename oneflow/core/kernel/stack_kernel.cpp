#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class StackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StackKernel);
  StackKernel() = default;
  ~StackKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    Blob* out_blob = BnInOp2Blob("out");
    // const int64_t lod_levels = out_blob->blob_desc().num_of_lod_levels();
    int64_t axis = this->op_conf().stack_conf().axis();
    // if (lod_levels > 1) {
    //   axis -= lod_levels - 1;
    // }
    const int64_t out_cols = out_blob->shape().Count(axis);
    const int64_t rows = out_blob->shape().elem_cnt() / out_cols;
    int64_t out_col_offset = 0;
    for (const auto& input_bn : this->op_attribute().input_bns()) {
      const Blob* in_blob = BnInOp2Blob(input_bn);
      const int64_t in_cols = in_blob->shape().Count(axis);
      CHECK_EQ(in_blob->shape().elem_cnt(), rows * in_cols);
      KernelUtil<device_type, T>::CopyColsRegion(ctx.device_ctx, rows, in_cols, in_blob->dptr<T>(),
                                                 0, in_cols, out_blob->mut_dptr<T>(),
                                                 out_col_offset, out_cols);
      out_col_offset += in_cols;
    }
    CHECK_LE(out_col_offset, out_cols);
  }
};

#define REGISTER_STACK_KERNEL(device, dtype)                                     \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kStackConf, device, dtype, \
                                        StackKernel<device, dtype>)

REGISTER_STACK_KERNEL(DeviceType::kGPU, float);
REGISTER_STACK_KERNEL(DeviceType::kGPU, double);
REGISTER_STACK_KERNEL(DeviceType::kCPU, float);
REGISTER_STACK_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
