#ifndef ONEFLOW_CORE_OPERATOR_BASIC_GRU_OP_H_
#define ONEFLOW_CORE_OPERATOR_BASIC_GRU_OP_H_

#include "oneflow/core/operator/recurrent_op.h"

namespace oneflow {

class BasicGruOp final : public RecurrentOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicGruOp);
  BasicGruOp() = default;
  ~BasicGruOp() = default;

  const PbMessage& GetCustomizedConf() const override;

 private:
  void VirtualInitFromOpConf();
  void VirtualIniferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BASIC_GRU_OP_H_
