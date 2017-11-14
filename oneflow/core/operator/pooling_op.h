#ifndef ONEFLOW_CORE_OPERATOR_POOLING_OP_H_
#define ONEFLOW_CORE_OPERATOR_POOLING_OP_H_

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class PoolingOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingOp);
  PoolingOp() = default;
  ~PoolingOp() = default;

  bool IsElemWise() const override { return true; }

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;

  void InferBlobDesc4FwBlobs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POOLING_OP_H_
