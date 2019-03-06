#ifndef ONEFLOW_CORE_OPERATOR_CONV_1D_V2_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_1D_V2_OP_H_

#include "oneflow/core/operator/conv_v2_op.h"

namespace oneflow {

class Conv1DV2Op final : public ConvV2Op<1> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv1DV2Op);
  Conv1DV2Op() = default;
  ~Conv1DV2Op() = default;

  const PbMessage& GetCustomizedConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_1D_V2_OP_H_
