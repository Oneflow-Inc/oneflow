#ifndef ONEFLOW_KERNEL_KERNEL_H_
#define ONEFLOW_KERNEL_KERNEL_H_

#include <memory>
#include <functional>
#include "job/resource.pb.h"
#include "job/job_conf.pb.h"
#include "register/blob.h"
#include "operator/operator.h"
#include "operator/operator_manager.h"
#include "operator/operator.pb.h"

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
  virtual void Forward(std::function<Blob*(const std::string& )>) = 0;
  virtual void Backward(std::function<Blob*(const std::string& )>) = 0;
 protected:
  Kernel() = default;
 private:
  std::unique_ptr<const Operator> op_;
};

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
