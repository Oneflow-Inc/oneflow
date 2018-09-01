#ifndef ONEFLOW_CORE_OPERATOR_COPY_LOCAL_OP_H_
#define ONEFLOW_CORE_OPERATOR_COPY_LOCAL_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CopyLocalOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyLocalOp);
  CopyLocalOp() = default;
  ~CopyLocalOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return GenPackedLbi(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_COPY_LOCAL_OP_H_
