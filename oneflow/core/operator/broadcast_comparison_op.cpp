#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

class BroadcastEqualOp final : public BroadcastBinaryOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastEqualOp);
  BroadcastEqualOp() = default;
  ~BroadcastEqualOp() = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().broadcast_equal_conf(); }
};

REGISTER_OP(OperatorConf::kBroadcastEqualConf, BroadcastEqualOp);

#define MAKE_BROADCAST_COMPARISION_OP_CLASS(class_name, op_name)                                 \
  class class_name##Op final : public BroadcastBinaryOp {                                        \
   public:                                                                                       \
    OF_DISALLOW_COPY_AND_MOVE(class_name##Op);                                                   \
    class_name##Op() = default;                                                                  \
    ~##class_name##Op() = default;                                                               \
                                                                                                 \
    const PbMessage& GetCustomizedConf() const override { return op_conf().##op_name##_conf(); } \
  };                                                                                             \
  REGISTER_OP(OperatorConf::k##class_name##Conf, class_name##Op);

}  // namespace oneflow
