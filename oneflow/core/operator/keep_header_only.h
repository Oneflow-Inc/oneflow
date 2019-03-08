#ifndef ONEFLOW_CORE_OPERATOR_KEEP_HEADER_ONLY_OP_H_
#define ONEFLOW_CORE_OPERATOR_KEEP_HEADER_ONLY_OP_H_

#include "oneflow/core/operator/identity_op.h"

namespace oneflow {

class KeepHeaderOnlyOp final : public IdentityOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeepHeaderOnlyOp);
  KeepHeaderOnlyOp() = default;
  ~KeepHeaderOnlyOp() override = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().keep_header_only_conf(); }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_KEEP_HEADER_ONLY_OP_H_
