#ifndef ONEFLOW_CORE_OPERATOR_RECORD_OP_H_
#define ONEFLOW_CORE_OPERATOR_RECORD_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class RecordOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordOp);
  RecordOp() = default;
  ~RecordOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;
  bool IsRecordOp() const override { return true; }

  void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) const override {}

 private:
  std::string ibn2lbn(const std::string& ibn) const override {
    return ibn2lbn_.at(ibn);
  }

  HashMap<std::string, std::string> ibn2lbn_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RECORD_OP_H_
