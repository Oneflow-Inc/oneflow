#ifndef ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_

#include "oneflow/core/operator/conv_base_op.h"

namespace oneflow {

class Conv2DOp : public ConvBaseOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv2DOp);
  Conv2DOp() = default;
  ~Conv2DOp() = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().conv_2d_conf();
  }

  int32_t ModelSplitAxis() const override {
    if (GetStringFromCustomizedConf("data_format") == "channel_first") {
      return 1;
    } else {
      return 3;
    }
  }
  int32_t MaxModelSplitNum() const override {
    return op_conf().conv_2d_conf().filters();
  }

 private:
  PbMessage* MutableCustomizedKernelConf(
      KernelConf* kernel_conf) const override {
    return kernel_conf->mutable_conv_2d_conf();
  }
  int32_t KernelDimSize() const override { return 2; }
};

REGISTER_OP(OperatorConf::kConv2DConf, Conv2DOp);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_
