#ifndef ONEFLOW_CORE_OPERATOR_CONV_3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_3D_OP_H_

#include "oneflow/core/operator/conv_op.h"

namespace oneflow {

class Conv3DOp final : public ConvOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv3DOp);
  Conv3DOp() = default;
  ~Conv3DOp() = default;

  const PbMessage& GetCustomizedConf() const override {
    return op_conf().conv_3d_conf();
  }

  int32_t ModelSplitAxis() const override {
    if (GetStringFromCustomizedConf("data_format") == "channels_first") {
      return 1;
    } else {
      return 4;
    }
  }
  int32_t MaxModelSplitNum() const override {
    return op_conf().conv_3d_conf().filters();
  }

 private:
  int32_t KernelDimSize() const override { return 3; }
};

REGISTER_OP(OperatorConf::kConv3DConf, Conv3DOp);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_3D_OP_H_
