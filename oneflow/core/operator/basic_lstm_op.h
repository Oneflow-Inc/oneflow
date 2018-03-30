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
  void VirtualInitFromOpConf();
  void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;

  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  //  ONEFLOW_CORE_OPERATOR_BASIC_LSTM_OP_H_
