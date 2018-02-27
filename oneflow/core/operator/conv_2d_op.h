#ifndef ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_

#include "oneflow/core/operator/conv_base_op.h"

namespace oneflow {

class Conv2DOp : public ConvBaseOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv2DOp);
  Conv2DOp() = default;
  ~Conv2DOp() = default;

  const PbMessage& GetSpecialConf() const override;

  int32_t ModelSplitAxis() const override;
  int32_t MaxModelSplitNum() const override {
    return op_conf().conv_2d_conf().filters();
  }

 private:
  virtual PbMessage* MutableConvKernelConf(KernelConf* kernel_conf) = 0;
  const int32_t kDimSize = 2;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_2D_OP_H_
