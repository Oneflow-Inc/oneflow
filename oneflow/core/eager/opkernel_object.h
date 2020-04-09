#ifndef ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_
#define ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/operator/user_op.h"
#include "oneflow/core/kernel/eager_kernel.h"

namespace oneflow {

class KernelCtx;
class Blob;

namespace eager {

class OpKernelObject : public vm::Object {
 public:
  OpKernelObject(const OpKernelObject&) = delete;
  OpKernelObject(OpKernelObject&&) = delete;
  OpKernelObject(const OperatorConf& op_conf, const std::shared_ptr<const JobDesc>& job_desc,
                 DeviceType device_type)
      : op_conf_(op_conf),
        job_desc_(job_desc),
        device_type_(device_type),
        kernel_(nullptr),
        opkernel_state_(nullptr) {
    CHECK(op_conf.has_user_conf());
  }
  ~OpKernelObject() override = default;

  UserOpConf* mut_user_op_conf() { return op_conf_.mutable_user_conf(); }

  const std::shared_ptr<user_op::OpKernelState>& opkernel_state() const { return opkernel_state_; }

  EagerKernel* mut_kernel() { return kernel_.get(); }
  void reset_opkernel_state(const std::shared_ptr<user_op::OpKernelState>& opkernel_state) {
    opkernel_state_ = opkernel_state;
  }

  void ResetOpAndKernel(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp);

 private:
  void InferBlobDescs(const Operator& op,
                      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, std::unique_ptr<OpContext>* op_ctx);
  void NewPartialInitializedKernel(
      const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, OpContext* op_ctx);

  OperatorConf op_conf_;
  std::shared_ptr<const JobDesc> job_desc_;
  DeviceType device_type_;
  std::unique_ptr<EagerKernel> kernel_;
  std::shared_ptr<user_op::OpKernelState> opkernel_state_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_
