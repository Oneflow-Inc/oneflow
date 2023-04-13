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
#include "oneflow/core/framework/user_op_tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/ep/include/stream.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/ep/include/stream.h"

namespace oneflow {

class JobDesc;

namespace user_op {

class KernelInitContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelInitContext);
  virtual ~KernelInitContext() = default;

  virtual ep::Stream* stream() = 0;

  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual const TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string&,
                                                              int32_t) const = 0;
  virtual const ParallelDesc& parallel_desc() const = 0;
  virtual const NdSbp& NdSbp4ArgNameAndIndex(const std::string&, int32_t) const = 0;

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
  const OperatorConf& op_conf() const { return user_op_conf().op_conf(); }

  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

  template<typename T>
  const T& attr(const std::string& attr_name) const;

 protected:
  KernelInitContext() = default;

  virtual const UserOpConfWrapper& user_op_conf() const = 0;
  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class KernelCacheContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelCacheContext);
  virtual ~KernelCacheContext() = default;

  virtual ep::Stream* stream() = 0;

  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual const TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string&,
                                                              int32_t) const = 0;
  virtual const ParallelDesc& parallel_desc() const = 0;
  virtual const NdSbp& NdSbp4ArgNameAndIndex(const std::string&, int32_t) const = 0;

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
  const OperatorConf& op_conf() const { return user_op_conf().op_conf(); }

  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

  template<typename T>
  const T& attr(const std::string& attr_name) const;

 protected:
  KernelCacheContext() = default;

  virtual const UserOpConfWrapper& user_op_conf() const = 0;
  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class KernelInferContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelInferContext);
  virtual ~KernelInferContext() = default;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;
  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;

  virtual ep::Stream* stream() = 0;
  virtual Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) = 0;
  virtual ShapeView ShapeView4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) = 0;
  virtual MutShapeView MutShapeView4ArgNameAndIndex(const std::string& arg_name,
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

  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

  virtual InferContext* MutOpInferContext() {
    UNIMPLEMENTED();
    return nullptr;
  }
  virtual const TensorDescInferFn& GetOpInferFn() const {
    UNIMPLEMENTED();
    static TensorDescInferFn empty_fn;
    return empty_fn;
  }

 protected:
  KernelInferContext() = default;

  virtual const UserOpConfWrapper& user_op_conf() const = 0;
  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class Tensor;

class KernelComputeContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelComputeContext);
  virtual ~KernelComputeContext() = default;

  virtual Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) = 0;
  virtual ep::Stream* stream() = 0;

  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                       int32_t index) const = 0;
  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;

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

  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

 protected:
  KernelComputeContext() = default;

  virtual const UserOpConfWrapper& user_op_conf() const = 0;

  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

class OpKernelState {
 public:
  virtual ~OpKernelState() = default;

 protected:
  OpKernelState() = default;
};

class OpKernelCache {
 public:
  virtual ~OpKernelCache() = default;

  static const int32_t kAllMayChanged = 0;
  static const int32_t kShapeNotChanged = 1 << 0;
  static const int32_t kAttrNotChanged = 1 << 1;

 protected:
  OpKernelCache() = default;
};

class OpKernel;

template<typename T, typename... Args>
OpKernel* NewOpKernel(Args&&... args);

class OpKernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpKernel);
  virtual ~OpKernel() = default;

  virtual std::shared_ptr<OpKernelState> CreateOpKernelState(KernelInitContext* ctx) const {
    return std::shared_ptr<OpKernelState>();
  }

  virtual std::shared_ptr<OpKernelCache> InitOpKernelCache(KernelCacheContext* ctx) const {
    return std::shared_ptr<OpKernelCache>();
  }

  virtual void InitOpKernelCacheWithFlags(KernelCacheContext* ctx, int8_t flag,
                                          std::shared_ptr<OpKernelCache>* cache_ptr) const {
    *cache_ptr = InitOpKernelCache(ctx);
  }

  virtual void Compute(KernelComputeContext* ctx, OpKernelState*, const OpKernelCache*) const {
    Compute(ctx);
  }
  virtual void Compute(KernelComputeContext* ctx) const {
    LOG(WARNING) << ctx->op_name() << " :UNIMPLEMENTED";
  }
  virtual void InferShape(KernelInferContext* ctx) const;
  virtual bool AlwaysComputeWhenAllOutputsEmpty() const = 0;
  virtual bool IsKernelLaunchSynchronized() const { return true; }

  bool has_state_or_cache() const { return has_state_or_cache_; }

 protected:
  OpKernel() : has_state_or_cache_(true) {}

 private:
  template<typename T, typename... Args>
  friend OpKernel* NewOpKernel(Args&&... args);
  bool has_state_or_cache_;
};

template<typename T, typename... Args>
OpKernel* NewOpKernel(Args&&... args) {
  OpKernel* ptr = new T(std::forward<Args>(args)...);
  ptr->has_state_or_cache_ = !(std::is_same<decltype(&OpKernel::CreateOpKernelState),
                                            decltype(&T::CreateOpKernelState)>::value
                               && std::is_same<decltype(&OpKernel::InitOpKernelCache),
                                               decltype(&T::InitOpKernelCache)>::value
                               && std::is_same<decltype(&OpKernel::InitOpKernelCacheWithFlags),
                                               decltype(&T::InitOpKernelCacheWithFlags)>::value);
  return ptr;
}

}  // namespace user_op

}  // namespace oneflow

#endif
