#ifndef ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_
#define ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class CopyCommNetOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetOp);
  CopyCommNetOp() = default;
  ~CopyCommNetOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_
