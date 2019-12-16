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
    int64_t axis = this->op_conf().stack_conf().axis();
    const int64_t out_cols = out_blob->shape().Count(axis);
    const int64_t rows = out_blob->shape().elem_cnt() / out_cols;
    CHECK_GT(rows, 0);
    int64_t out_col_offset = 0;
    for (const auto& input_bn : this->op_attribute().input_bns()) {
      const Blob* in_blob = BnInOp2Blob(input_bn);
      const int64_t in_cols = in_blob->shape().Count(axis);
      CHECK_EQ(in_blob->shape().elem_cnt(), rows * in_cols);
      if (rows * in_cols > 0) {
        KernelUtil<device_type, T>::CopyColsRegion(
            ctx.device_ctx, rows, in_cols, in_blob->dptr<T>(), 0, in_cols, out_blob->mut_dptr<T>(),
            out_col_offset, out_cols);
      }
      out_col_offset += in_cols;
    }
    CHECK_LE(out_col_offset, out_cols);
  }

  void ForwardShape(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_0 = BnInOp2Blob(this->op_attribute().input_bns().Get(0));
    DimVector shape_vec;
    in_0->shape().ToDimVector(&shape_vec);
    int64_t stack_axis = this->op_conf().stack_conf().axis();

    FOR_RANGE(size_t, i, 1, this->op_attribute().input_bns().size()) {
      const Blob* in_i = BnInOp2Blob(this->op_attribute().input_bns().Get(i));
      CHECK_EQ(in_i->shape().NumAxes(), shape_vec.size());
      FOR_RANGE(int64_t, j, 0, in_i->shape().NumAxes()) {
        if (j == stack_axis) {
          shape_vec.at(j) += in_i->shape().At(j);
        } else {
          CHECK_EQ(in_i->shape().At(j), shape_vec.at(j));
        }
      }
    }
    BnInOp2Blob("out")->mut_shape_view()->set_shape(Shape(std::move(shape_vec)));
  }
};

template<DeviceType device_type, typename T>
class StackGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StackGradKernel);
  StackGradKernel() = default;
  ~StackGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    int64_t axis = this->op_conf().stack_grad_conf().axis();

    const int64_t in_cols = in_blob->shape().Count(axis);
    const int64_t rows = in_blob->shape().elem_cnt() / in_cols;
    CHECK_GT(rows, 0);
    int64_t in_col_offset = 0;
    for (const auto& obn : this->op_attribute().output_bns()) {
      Blob* out_blob = BnInOp2Blob(obn);
      const int64_t out_cols = out_blob->shape().Count(axis);
      CHECK_EQ(out_blob->shape().elem_cnt(), rows * out_cols);
      if (rows * out_cols > 0) {
        KernelUtil<device_type, T>::CopyColsRegion(ctx.device_ctx, rows, out_cols,
                                                   in_blob->dptr<T>(), in_col_offset, in_cols,
                                                   out_blob->mut_dptr<T>(), 0, out_cols);
      }
      in_col_offset += out_cols;
      CHECK_LE(in_col_offset, in_cols);
    }
  }
};

#define REGISTER_STACK_KERNEL(device, dtype)                                         \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kStackConf, device, dtype,     \
                                        StackKernel<device, dtype>)                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kStackGradConf, device, dtype, \
                                        StackGradKernel<device, dtype>)

REGISTER_STACK_KERNEL(DeviceType::kGPU, float);
REGISTER_STACK_KERNEL(DeviceType::kGPU, double);
REGISTER_STACK_KERNEL(DeviceType::kGPU, int32_t);
REGISTER_STACK_KERNEL(DeviceType::kGPU, int8_t);
REGISTER_STACK_KERNEL(DeviceType::kCPU, float);
REGISTER_STACK_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
