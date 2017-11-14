#ifndef ONEFLOW_CORE_OPERATOR_RECORD_OP_H_
#define ONEFLOW_CORE_OPERATOR_RECORD_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class RecordOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordOp);
  RecordOp() = default;
  ~RecordOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  bool IsRecordOp() const override { return true; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) override {}

 private:
  std::string ibn2lbn(const std::string& ibn) const override {
    return ibn2lbn_.at(ibn);
  }

  HashMap<std::string, std::string> ibn2lbn_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RECORD_OP_H_
