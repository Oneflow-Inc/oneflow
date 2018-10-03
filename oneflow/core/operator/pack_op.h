#ifndef ONEFLOW_CORE_OPERATOR_PACK_OP_H_
#define ONEFLOW_CORE_OPERATOR_PACK_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PackOp final : public Operator {
 public:
  OF_DISALLOW_COPY(PackOp);
  PackOp() = default;
  ~PackOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().pack_conf(); }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PACK_OP_H_
