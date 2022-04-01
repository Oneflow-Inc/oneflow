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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_KERNEL_H_

#include <memory>

#include <glog/logging.h>

#include "oneflow/core/framework/util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

class JobDesc;

namespace user_op {

class KernelCreateContext {
 public:
  virtual ~KernelCreateContext() = default;

  const std::string& input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const { return user_op_conf().op_name(); }
  const std::string& op_type_name() const { return user_op_conf().op_type_name(); }
  const std::string& device_tag() const { return user_op_conf().op_conf().device_tag(); }
  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

 protected:
  virtual const UserOpConfWrapper& user_op_conf() const = 0;
  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class KernelInitContext {
 public:
  virtual ~KernelInitContext() = default;

  virtual DeviceCtx* device_ctx() = 0;

  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual const cfg::SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&,
                                                              int32_t) const = 0;
  virtual const TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string&,
                                                              int32_t) const = 0;
  virtual const ParallelDesc& parallel_desc() const = 0;
  virtual const cfg::ParallelDistribution& ParallelDistribution4ArgNameAndIndex(const std::string&,
                                                                                int32_t) const = 0;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  const std::string& input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const { return user_op_conf().op_name(); }
  const std::string& op_type_name() const { return user_op_conf().op_type_name(); }
  const std::string& device_tag() const { return user_op_conf().op_conf().device_tag(); }
  const OperatorConf& op_conf() const { return user_op_conf().op_conf(); }

  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

  template<typename T>
  const T& attr(const std::string& attr_name) const;

 protected:
  KernelInitContext() = default;
  KernelInitContext(const KernelInitContext&) = delete;

  virtual const UserOpConfWrapper& user_op_conf() const = 0;
  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class KernelInferContext {
 public:
  virtual ~KernelInferContext() = default;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;

  virtual DeviceCtx* device_ctx() = 0;
  virtual Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) = 0;
  virtual const ShapeView& ShapeView4ArgNameAndIndex(const std::string& arg_name,
                                                     int32_t arg_index) = 0;
  virtual MutShapeView* MutShapeView4ArgNameAndIndex(const std::string& arg_name,
                                                     int32_t arg_index) = 0;

  const std::string& input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const { return user_op_conf().op_name(); }
  const std::string& op_type_name() const { return user_op_conf().op_type_name(); }
  const std::string& device_tag() const { return user_op_conf().op_conf().device_tag(); }

  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

  virtual InferContext* MutOpInferContext() {
    UNIMPLEMENTED();
    return nullptr;
  }
  virtual const TensorDescInferFn& GetOpInferFn() const { UNIMPLEMENTED(); }

 protected:
  KernelInferContext() = default;
  KernelInferContext(const KernelInferContext&) = delete;

  virtual const UserOpConfWrapper& user_op_conf() const = 0;
  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class Tensor;

class KernelComputeContext {
 public:
  virtual ~KernelComputeContext() = default;

  virtual Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) = 0;
  virtual DeviceCtx* device_ctx() = 0;

  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                       int32_t index) const = 0;
  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const JobDesc& job_desc() const = 0;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;
  const std::string& input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().output(arg_name, index);
  }
  bool has_input(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const {
    return user_op_conf().has_output(arg_name, index);
  }
  int32_t input_size(const std::string& arg_name) const {
    return user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const {
    return user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const { return user_op_conf().op_name(); }
  const std::string& op_type_name() const { return user_op_conf().op_type_name(); }
  const std::string& device_tag() const { return user_op_conf().op_conf().device_tag(); }

  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

 protected:
  KernelComputeContext() = default;
  KernelComputeContext(const KernelComputeContext&) = delete;

  virtual const UserOpConfWrapper& user_op_conf() const = 0;

  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class OpKernelState {
 public:
  virtual ~OpKernelState() = default;

 protected:
  OpKernelState() = default;
};

class OpKernel;

template<typename T>
OpKernel* NewOpKernel();

enum OpKernelStatefulness {
  kInvalidOpKernelStatefulness = 0,
  kStatefulOpKernel,
  kStatelessOpKernel,
};

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  bool is_stateless() const {
    CHECK_NE(statefullness_, kInvalidOpKernelStatefulness);
    return statefullness_ == kStatelessOpKernel;
  }

  virtual std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const {
    return std::shared_ptr<OpKernelState>();
  }

  virtual void Compute(KernelComputeContext* ctx, OpKernelState*) const { Compute(ctx); }
  virtual void Compute(KernelComputeContext*) const { LOG(INFO) << "UNIMPLEMENTED"; }
  virtual void InferShape(KernelInferContext* ctx) const;
  virtual bool AlwaysComputeWhenAllOutputsEmpty() const = 0;
  virtual bool IsKernelLaunchSynchronized() const { return true; }

 protected:
  OpKernel() : statefullness_(kInvalidOpKernelStatefulness) {}

 private:
  template<typename T>
  friend OpKernel* NewOpKernel();
  template<typename T>
  friend OpKernel* NewOpKernelWithCtx(KernelCreateContext* ctx);
  OpKernelStatefulness statefullness_;
};

template<typename T>
OpKernel* NewOpKernel() {
  OpKernel* opkernel = new T();
  if (typeid(&OpKernel::CreateOpKernelState) == typeid(&T::CreateOpKernelState)) {
    opkernel->statefullness_ = kStatelessOpKernel;
  } else {
    opkernel->statefullness_ = kStatefulOpKernel;
  }
  return opkernel;
}

class KernelCreateContext;

template<typename T>
OpKernel* NewOpKernelWithCtx(KernelCreateContext* ctx) {
  OpKernel* opkernel = new T(ctx);
  if (typeid(&OpKernel::CreateOpKernelState) == typeid(&T::CreateOpKernelState)) {
    opkernel->statefullness_ = kStatelessOpKernel;
  } else {
    opkernel->statefullness_ = kStatefulOpKernel;
  }
  return opkernel;
}

}  // namespace user_op

}  // namespace oneflow

#endif
