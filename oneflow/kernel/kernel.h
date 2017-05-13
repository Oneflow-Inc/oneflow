#ifndef ONEFLOW_KERNEL_KERNEL_H_
#define ONEFLOW_KERNEL_KERNEL_H_

#include <memory>
#include <functional>
#include "operator/operator.h"
#include "operator/operator.pb.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel() = default;

  virtual void InitFromOpProto(const OperatorProto& op_proto) = 0;
  virtual void Forward(std::function<void()>) = 0;
  virtual void Backward(std::function<void()>) = 0;
 protected:
  Kernel() = default;
 private:
  std::unique_ptr<Operator> op_;
};

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_KERNEL_H_
