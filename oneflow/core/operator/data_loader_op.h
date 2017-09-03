#ifndef ONEFLOW_CORE_OPERATOR_DATA_LOADER_OP_H_
#define ONEFLOW_CORE_OPERATOR_DATA_LOADER_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class DataLoaderOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataLoaderOp);
  DataLoaderOp() = default;
  ~DataLoaderOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;

  void InferBlobDesc4FwBlobs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) override;

 private:
  std::string obn2lbn(const std::string& output_bn) const override {
    return op_name() + "/"
           + GetMsgFromSpecialConf<LogicalBlob>(output_bn).name();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DATA_LOADER_OP_H_
