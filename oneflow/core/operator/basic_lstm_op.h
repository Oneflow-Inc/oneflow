#ifdef ONEFLOW_CORE_OPERATOR_BASIC_LSTM_OP_H_
#define ONEFLOW_CORE_OPERATOR_BASIC_LSTM_OP_H_

#include "oneflow/core/operator/recurrent_op.h"
#include "oneflwo/core/operator/operator.h"

namespace oneflow {

class BasicLstmOp final : public RecurrentOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicLstmOp);
  BasicLstmOp() = default;
  ~BasicLstmOp() = default;
  const PbMessage& GetCustomizedConf() const override;

 private:
  void VirtualInitFromOpConf();
  void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BASIC_LSTM_OP_H_
