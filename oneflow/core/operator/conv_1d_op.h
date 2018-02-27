#ifndef ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_

#include "oneflow/core/operator/conv_base_op.h"

namespace oneflow {

class Conv1DOp : public ConvBaseOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv1DOp);
  Conv1DOp() = default;
  ~Conv1DOp() = default;

  void InitFromOpConf() override;

  const PbMessage& GetSpecialConf() const override;

  int32_t ModelSplitAxis() const override;
  int32_t MaxModelSplitNum() const override {
    return op_conf().conv_1d_conf().filters();
  }

 private:
  PbMessage* MutableConvKernelConf(KernelConf* kernel_conf) override;
  const int32_t kDimSize = 1;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_
