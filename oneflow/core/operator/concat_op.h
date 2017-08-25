#ifndef ONEFLOW_CORE_OPERATOR_CONCAT_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONCAT_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class ConcatOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatOp);
  ConcatOp() = default;
  ~ConcatOp() = default;

  void InitFromOpConf() override;

  const PbMessage& GetSpecialConf() const override;

  void InferBlobDesc4FwBlobs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return ibn2lbn_.at(input_bn);
  }

  std::unordered_map<std::string, std::string> ibn2lbn_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONCAT_OP_H_
