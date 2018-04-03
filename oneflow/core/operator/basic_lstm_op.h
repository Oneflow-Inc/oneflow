#ifndef ONEFLOW_CORE_OPERATOR_BASIC_LSTM_OP_H_
#define ONEFLOW_CORE_OPERATOR_BASIC_LSTM_OP_H_

#include "oneflow/core/operator/recurrent_op.h"

namespace oneflow {

class BasicLstmOp final : public RecurrentOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicLstmOp);
  BasicLstmOp() = default;
  ~BasicLstmOp() = default;
  const PbMessage& GetSpecialConf() const override;

 private:
  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

  std::string VirtualIbn2Lbn(const std::string& input_bn) const override;
  std::string VirtualObn2Lbn(const std::string& output_bn) const override;

  void InitCellFromOpConf();
};

}  // namespace oneflow

#endif  //  ONEFLOW_CORE_OPERATOR_BASIC_LSTM_OP_H_
