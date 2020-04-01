#ifndef ONEFLOW_CORE_EAGER_OP_OBJECT_H_
#define ONEFLOW_CORE_EAGER_OP_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {
namespace eager {

class OpObject : public vm::Object {
 public:
  OpObject(const OpObject&) = delete;
  OpObject(OpObject&&) = delete;

  OpObject(const OperatorConf& op_conf, const JobDesc* job_desc)
      : op_(ConstructOp(op_conf, job_desc)) {}
  ~OpObject() override = default;

  const Operator& op() const { return *op_; }

 private:
  std::shared_ptr<Operator> op_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_OP_OBJECT_H_
