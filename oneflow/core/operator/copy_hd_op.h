#ifndef ONEFLOW_CORE_OPERATOR_COPY_HD_OP_H_
#define ONEFLOW_CORE_OPERATOR_COPY_HD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CopyHdOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdOp);
  CopyHdOp() = default;
  ~CopyHdOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override {
    if (this->op_conf().copy_hd_conf().has_lbi()) {
      return this->op_conf().copy_hd_conf().lbi();
    } else {
      return GenPackedLbi();
    }
  }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override {
    if (this->op_conf().copy_hd_conf().has_lbi()) {
      return this->op_conf().copy_hd_conf().lbi();
    } else {
      return GenPackedLbi();
    }
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_COPY_HD_OP_H_
