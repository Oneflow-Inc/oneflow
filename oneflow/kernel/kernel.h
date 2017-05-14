#ifndef ONEFLOW_KERNEL_KERNEL_H_
#define ONEFLOW_KERNEL_KERNEL_H_

#include <memory>
#include <functional>
#include "register/blob.h"
#include "operator/operator.h"
#include "operator/operator_manager.h"
#include "operator/operator.pb.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  Kernel() = default;
  virtual ~Kernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) {
    Operator* op = CreateOp(op_proto.op_conf().op_type_case());
    op->InitFromProto(op_proto);
    op_.reset(op);
  }
  virtual void Forward(
      std::function<Blob*(const std::string& bn_in_op)>) = 0;
  virtual void Backward(
      std::function<Blob*(const std::string& bn_in_op)>) = 0;
 private:
  std::unique_ptr<const Operator> op_;
};

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_KERNEL_H_
