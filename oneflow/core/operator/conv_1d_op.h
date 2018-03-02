#ifndef ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_

#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

class Conv1DOp final : public ConvOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv1DOp);
  Conv1DOp() = default;
  ~Conv1DOp() = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().conv_1d_conf();
  }

  int32_t ModelSplitAxis() const override {
    if (GetStringFromCustomizedConf("data_format") == "channels_first") {
      return 1;
    } else {
      return 2;
    }
  }
  int32_t MaxModelSplitNum() const override {
    return op_conf().conv_1d_conf().filters();
  }

 private:
  int32_t KernelDimSize() const override { return 1; }
};

REGISTER_OP(OperatorConf::kConv1DConf, Conv1DOp);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_
