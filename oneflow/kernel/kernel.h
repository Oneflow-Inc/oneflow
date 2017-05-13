#ifndef ONEFLOW_KERNEL_KERNEL_H_
#define ONEFLOW_KERNEL_KERNEL_H_

#include <memory>
#include <functional>
#include "register/blob.h"
#include "operator/operator.h"
#include "operator/operator.pb.h"

namespace oneflow {

template<typename Dtype>
class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  Kernel() = default;
  virtual ~Kernel() = default;

  virtual void InitFromOpProto(const OperatorProto& op_proto) = 0;
  virtual void Forward(
      std::function<Blob<Dtype>*(const std::string bn_in_op)>) = 0;
  virtual void Backward(
      std::function<Blob<Dtype>*(const std::string bn_in_op)>) = 0;
 private:
  std::unique_ptr<Operator> op_;
};

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_KERNEL_H_
