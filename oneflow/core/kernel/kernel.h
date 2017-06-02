#ifndef ONEFLOW_KERNEL_KERNEL_H_
#define ONEFLOW_KERNEL_KERNEL_H_

#include <memory>
#include <functional>
#include "oneflow/core/conf/resource.pb.h"
#include "oneflow/core/conf/job_conf.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/operator_manager.h"
#include "oneflow/core/operator/operator.pb.h"

namespace oneflow {

class Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Kernel);
  virtual ~Kernel() = default;

  void InitFromOpProto(const OperatorProto& op_proto) {
    Operator* op = CreateOp(op_proto.op_conf().op_type_case());
    op->InitFromProto(op_proto);
    op_.reset(op);
  }
  // for Forward / Bp Calculation in FwExecGragh node and BpExecGragh node
  // through bn_in_op2blob_ptr function get the input blob and output blob
  // the Kernel will using the input blob calculate the result and fill output
  virtual void Forward(std::function<Blob*(const std::string&)>) const = 0;
  virtual void Backward(std::function<Blob*(const std::string&)>) const = 0;

  //
  const std::string& GetLbnFromBnInOp(const std::string& bn_in_op) const {
    return op_->Lbn4BnInOp(bn_in_op);
  }

 protected:
  Kernel() = default;
 private:
  std::unique_ptr<const Operator> op_;
};

using KernelWardFunc = void (Kernel::*)(std::function<Blob*(const std::string&)>) const;

#define INSTANTIATE_CPU_KERNEL_CLASS(classname) \
  char gInstantiationGuardCPU##classname; \
  template class classname<DeviceType::kCPU, FloatingPointType::kFloat>; \
  template class classname<DeviceType::kCPU, FloatingPointType::kDouble>;
#define INSTANTIATE_GPU_KERNEL_CLASS(classname) \
  char gInstantiationGuardGPU##classname; \
  template class classname<DeviceType::kGPU, FloatingPointType::kFloat>; \
  template class classname<DeviceType::kGPU, FloatingPointType::kDouble>;

}  // namespace oneflow

#endif  // ONEFLOW_KERNEL_KERNEL_H_
