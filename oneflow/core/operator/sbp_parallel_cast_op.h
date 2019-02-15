#ifndef ONEFLOW_CORE_OPERATOR_SBP_PARALLEL_CAST_OP_H_
#define ONEFLOW_CORE_OPERATOR_SBP_PARALLEL_CAST_OP_H_

#include "oneflow/core/operator/identity_op.h"

namespace oneflow {

class SbpParallelCastOp final : public IdentityOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SbpParallelCastOp);
  SbpParallelCastOp() = default;
  ~SbpParallelCastOp() override = default;

  const PbMessage& GetCustomizedConf() const override { return op_conf().sbp_parallel_cast_conf(); }

 private:
  void GetOpParallelSignatures(
      std::vector<std::unique_ptr<const OpParallelSignature>>*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SBP_PARALLEL_CAST_OP_H_
