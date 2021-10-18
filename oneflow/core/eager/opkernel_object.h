/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_
#define ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_

#include "oneflow/core/vm/object.h"
#include "oneflow/core/operator/user_op.h"
#include "oneflow/core/kernel/eager_kernel.h"
#include "oneflow/core/eager/blob_object.h"
#include "oneflow/core/operator/op_node_signature_desc.h"
#include "oneflow/core/stream/stream_context_adapter.h"

namespace oneflow {

class Blob;
class ParallelContext;

namespace vm {

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

  const JobDesc& job_desc() const { return *job_desc_; }

  const std::string& op_name() const { return op_conf_.name(); }
  UserOpConf* mut_user_op_conf() { return op_conf_.mutable_user_conf(); }

  const std::shared_ptr<user_op::OpKernelState>& opkernel_state() const { return opkernel_state_; }

  const EagerKernel& kernel() const { return *kernel_; }
  EagerKernel* mut_kernel() { return kernel_.get(); }
  void reset_opkernel_state(const std::shared_ptr<user_op::OpKernelState>& opkernel_state) {
    opkernel_state_ = opkernel_state;
  }

  Maybe<void> ResetOpAndKernel(const OpNodeSignatureDesc& op_node_signature,
                               const ParallelContext* parallel_ctx,
                               const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
                               const ParallelDesc* parallel_desc);

 private:
  Maybe<void> InferBlobDescs(const Operator& op,
                             const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
                             const cfg::SbpSignature* sbp_signature,
                             const ParallelContext* parallel_ctx);
  void NewPartialInitializedKernel(
      const Operator& op, const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
      const OpNodeSignatureDesc& op_node_signature, const ParallelContext* parallel_ctx,
      const ParallelDesc* parallel_desc);

  OperatorConf op_conf_;
  std::shared_ptr<const JobDesc> job_desc_;
  DeviceType device_type_;
  std::unique_ptr<EagerKernel> kernel_;
  std::shared_ptr<user_op::OpKernelState> opkernel_state_;
};

class SystemOpKernelContext : public KernelContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SystemOpKernelContext);
  explicit SystemOpKernelContext(DeviceCtx* device_ctx) : device_ctx_(device_ctx) {}
  ~SystemOpKernelContext() = default;

  StreamContext* stream_ctx() const override { return stream_ctx_.get(); }

  DeviceCtx* device_ctx() const override { return device_ctx_; }

  Blob* BnInOp2Blob(const std::string& bn) const override { return bn_in_op2blob_fn_(bn); }

  const std::shared_ptr<KernelState>& state() const override {
    static const std::shared_ptr<KernelState> null_state;
    return null_state;
  }

  void set_state(std::shared_ptr<KernelState> state) override { UNIMPLEMENTED(); }

  void set_device_ctx(DeviceCtx* ctx) {
    stream_ctx_.reset(NewStreamContextAdapter(ctx));
    device_ctx_ = ctx;
  }

  void UpdateBnInOp2BlobFn(std::function<Blob*(const std::string&)> fn) {
    bn_in_op2blob_fn_ = std::move(fn);
  }

 private:
  DeviceCtx* device_ctx_;
  std::function<Blob*(const std::string&)> bn_in_op2blob_fn_;
  std::unique_ptr<StreamContext> stream_ctx_;
};

class SystemOpKernelObject : public vm::Object {
 public:
  SystemOpKernelObject(const SystemOpKernelObject&) = delete;
  SystemOpKernelObject(SystemOpKernelObject&&) = delete;
  SystemOpKernelObject(const OperatorConf& op_conf, const std::shared_ptr<const JobDesc>& job_desc,
                       DeviceType device_type)
      : op_conf_(op_conf), job_desc_(job_desc), device_type_(device_type), kernel_(nullptr) {}
  ~SystemOpKernelObject() override = default;

  const JobDesc& job_desc() const { return *job_desc_; }

  const std::string& op_name() const { return op_conf_.name(); }
  const OperatorConf& op_conf() const { return op_conf_; }

  const Kernel& kernel() const { return *kernel_; }

  SystemOpKernelContext* kernel_ctx() const { return kernel_ctx_.get(); }

  Maybe<void> ResetKernel(const OpNodeSignatureDesc& op_node_signature,
                          const ParallelContext* parallel_ctx,
                          const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
                          const ParallelDesc* parallel_desc);

 private:
  Maybe<void> InferBlobDescs(const Operator& op,
                             const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
                             const cfg::SbpSignature* sbp_signature,
                             const ParallelContext* parallel_ctx);
  void ResetKernel(const Operator& op,
                   const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
                   const OpNodeSignatureDesc& op_node_signature,
                   const ParallelContext* parallel_ctx, const ParallelDesc* parallel_desc);

  OperatorConf op_conf_;
  std::shared_ptr<const JobDesc> job_desc_;
  DeviceType device_type_;
  std::unique_ptr<const Kernel> kernel_;
  std::unique_ptr<SystemOpKernelContext> kernel_ctx_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_OPKERNEL_OBJECT_H_
