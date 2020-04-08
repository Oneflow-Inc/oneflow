#ifndef ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_
#define ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/operator/user_op.h"
#include "oneflow/core/kernel/user_kernel.h"

namespace oneflow {

class KernelCtx;
class Blob;

namespace eager {

class OpKernelObject : public vm::Object {
 public:
  OpKernelObject(const OpKernelObject&) = delete;
  OpKernelObject(OpKernelObject&&) = delete;
  OpKernelObject(const UserOpConf& user_op_conf, const std::shared_ptr<const JobDesc>& job_desc)
      : user_op_conf_(user_op_conf),
        job_desc_(job_desc),
        op_(nullptr),
        kernel_(nullptr),
        opkernel_state_(nullptr) {}
  ~OpKernelObject() override = default;

  const UserOp& op() const { return *op_; }
  const std::shared_ptr<user_op::OpKernelState>& opkernel_state() const { return opkernel_state_; }

  UserKernel* mut_kernel() { return kernel_.get(); }
  void reset_opkernel_state(const std::shared_ptr<user_op::OpKernelState>& opkernel_state) {
    opkernel_state_ = opkernel_state;
  }

  void InferAndNewUninitiatedKernel(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp);

 private:
  void InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, std::unique_ptr<OpContext>* op_ctx);
  void NewPartialInitializedKernel(
      const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const ParallelContext* parallel_ctx, OpContext* op_ctx);

  UserOpConf user_op_conf_;
  std::shared_ptr<const JobDesc> job_desc_;
  std::shared_ptr<UserOp> op_;
  std::unique_ptr<UserKernel> kernel_;
  std::shared_ptr<user_op::OpKernelState> opkernel_state_;
};

}  // namespace eager
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_
