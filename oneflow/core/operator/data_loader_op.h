#ifndef OPERATOR_DATA_LOADER_OP_H_
#define OPERATOR_DATA_LOADER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class DataLoaderOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoaderOp);
  DataLoaderOp() = default;
  ~DataLoaderOp() = default;
  
  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;
  
  void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_num) const override;
  
 private:
  std::string obn2lbn(const std::string& output_bn) const override {
    return op_name() + "/" + GetStringFromSpecialConf(output_bn);
  }

};

} // namespace oneflow

#endif // OPERATOR_DATA_LOADER_OP_H_
