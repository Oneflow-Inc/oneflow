#ifndef ONEFLOW_CORE_OPERATOR_RNN_LOOKUP_OP_H_
#define ONEFLOW_CORE_OPERATOR_RNN_LOOKUP_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class RnnLookupOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnLookupOp);
  RnnLookupOp() = default;
  ~RnnLookupOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;

  void InferBlobDesc4FwBlobs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RNN_LOOKUP_OP_H_
