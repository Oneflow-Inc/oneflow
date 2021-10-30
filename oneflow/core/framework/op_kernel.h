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
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/user_op_registry.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_kernel_context_if.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/stream/stream_context.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

class JobDesc;

namespace user_op {

class KernelCreateContext : virtual public OpInfoIf, virtual public DeviceInfoIf, virtual public AttrIf {
 public:
  virtual ~KernelCreateContext() = default;
};

class KernelInitContext : virtual public UserOpConfOpInfoProvider,
  virtual public AttrIf,
                          virtual public InputAndOutputNameIf,
                          virtual public StreamCtxAndDeviceCtxIf,
                          virtual public ConsistentInfoIf,
                          virtual public TensorDescIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelInitContext);
  virtual ~KernelInitContext() = default;

  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const ParallelDesc& parallel_desc() const = 0;
  const OperatorConf& op_conf() const { return user_op_conf().op_conf(); }

 protected:
  KernelInitContext() = default;
};

class KernelInferContext : virtual public OpInfoIf, virtual public DeviceInfoIf,
                           virtual public InputAndOutputNameIf,
                           virtual public AttrIf,
                           virtual public StreamCtxAndDeviceCtxIf,
                           virtual public TensorDescIf,
                           virtual public TensorObjIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelInferContext);
  virtual ~KernelInferContext() = default;

  virtual const ParallelContext& parallel_ctx() const = 0;

  virtual const ShapeView& ShapeView4ArgNameAndIndex(const std::string& arg_name,
                                                     int32_t arg_index) = 0;
  virtual MutShapeView* MutShapeView4ArgNameAndIndex(const std::string& arg_name,
                                                     int32_t arg_index) = 0;

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
};

class Tensor;

class KernelComputeContext : virtual public OpInfoIf, virtual public DeviceInfoIf,
                             virtual public InputAndOutputNameIf,
                             virtual public AttrIf,
                             virtual public StreamCtxAndDeviceCtxIf,
                             virtual public TensorDescIf,
                             virtual public TensorObjIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelComputeContext);
  virtual ~KernelComputeContext() = default;

  virtual const ParallelContext& parallel_ctx() const = 0;

 protected:
  KernelComputeContext() = default;
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

  static const int8_t ShapeMayChanged = 0x1;
  static const int8_t AttrMayChanged = 0x10;

 protected:
  OpKernelCache() = default;
};

class OpKernel;

template<typename T>
OpKernel* NewOpKernel();

using KernelCacheContext = KernelInitContext;

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

  virtual void InitOpKernelCache(KernelCacheContext* ctx, int8_t flag,
                                 std::shared_ptr<OpKernelCache>* cache) const {
    *cache = InitOpKernelCache(ctx);
  }

  virtual void Compute(KernelComputeContext* ctx, OpKernelState*, const OpKernelCache*) const { Compute(ctx); }
  virtual void Compute(KernelComputeContext*) const { LOG(INFO) << "UNIMPLEMENTED"; }
  virtual void InferShape(KernelInferContext* ctx) const;
  virtual bool AlwaysComputeWhenAllOutputsEmpty() const = 0;

 protected:
  OpKernel() {}

 private:
  template<typename T>
  friend OpKernel* NewOpKernel();
  template<typename T>
  friend OpKernel* NewOpKernelWithCtx(KernelCreateContext* ctx);
};

template<typename T>
OpKernel* NewOpKernel() {
  return new T();
}

class KernelCreateContext;

template<typename T>
OpKernel* NewOpKernelWithCtx(KernelCreateContext* ctx) {
  return new T(ctx);
}

}  // namespace user_op

}  // namespace oneflow

#endif
