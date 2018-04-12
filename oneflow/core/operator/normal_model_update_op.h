#ifndef ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NormalModelUpdtOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalModelUpdtOp);
  virtual ~NormalModelUpdtOp() = default;

  virtual void InitFromOpConf();
  const PbMessage& GetCustomizedConf() const override;

 protected:
  NormalModelUpdtOp() = default;
  virtual void MdUpdtVirtualInitFromOpConf() {}

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMAL_MODEL_UPDATE_OP_H_
