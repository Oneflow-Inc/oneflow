#ifndef ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_

#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

class Conv1DOp final : public ConvOp<1> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv1DOp);
  Conv1DOp() = default;
  ~Conv1DOp() = default;

  const PbMessage& GetCustomizedConf() const override;
  ActivationType GetActivationType() const override;
  bool UseActivation() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_1D_OP_H_
