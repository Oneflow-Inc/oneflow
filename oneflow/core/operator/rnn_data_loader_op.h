#ifndef ONEFLOW_CORE_OPERATOR_RNN_DATA_LOADER_OP_H_
#define ONEFLOW_CORE_OPERATOR_RNN_DATA_LOADER_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class RnnDataLoaderOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnDataLoaderOp);
  RnnDataLoaderOp() = default;
  ~RnnDataLoaderOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;

  void InferBlobDesc4FwBlobs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) override;
  std::string obn2lbn(const std::string& output_bn) const override {
    return op_name() + "/" + GetStringFromSpecialConf(output_bn);
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RNN_DATA_LOADER_OP_H_
