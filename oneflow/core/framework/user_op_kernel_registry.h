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
#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_KERNEL_REGISTRY_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_KERNEL_REGISTRY_H_

#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/common/high_order_bool.h"

namespace oneflow {

namespace user_op {

class OpKernel;
class TensorDesc;
class InferContext;

class KernelRegContext {
 public:
  virtual ~KernelRegContext() = default;

  virtual DeviceType device_type() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;

  virtual const std::vector<std::pair<std::string, int32_t>>& inputs() const = 0;
  virtual const std::vector<std::pair<std::string, int32_t>>& outputs() const = 0;

  virtual const UserOpConfWrapper& user_op_conf() const = 0;

  template<typename T>
  const T& Attr(const std::string& attr_name) const {
    return AttrValueCast<T>(*Attr4Name(attr_name));
  }

 protected:
  KernelRegContext() = default;
  KernelRegContext(const KernelRegContext&) = delete;
  virtual const std::shared_ptr<const AttrVal>& Attr4Name(const std::string& attr_name) const = 0;
};

using OpKernelCreateFn = std::function<const OpKernel*()>;
using InferTmpSizeFn = std::function<size_t(InferContext*)>;
using AddInplaceArgPair = std::function<Maybe<void>(
    const std::string& out_arg_name, int32_t out_arg_index, const std::string& in_arg_name,
    int32_t in_arg_index, bool is_mutable)>;
using InplaceProposalFn = std::function<Maybe<void>(const InferContext&, AddInplaceArgPair)>;
using IsMatchedHob = std::shared_ptr<hob::BaseExpr<user_op::KernelRegContext, bool>>;

constexpr int kKernelPriorityFallback = -10;
constexpr int kKernelPriorityDefault = 0;
constexpr int kKernelPriorityOptimized = 10;
constexpr int kKernelPriorityExperimental = 100;

struct OpKernelRegistryResult {
  std::string op_type_name;

  OpKernelCreateFn create_fn;
  bool need_temp_storage;
  InferTmpSizeFn infer_tmp_size_fn;
  InplaceProposalFn inplace_proposal_fn;
  IsMatchedHob is_matched_hob;
  int32_t priority = kKernelPriorityDefault;
};

class OpKernelRegistry final {
 public:
  OpKernelRegistry& Name(const std::string& op_type_name);

  template<typename T>
  OpKernelRegistry& SetCreateFn() {
    return SetCreateFn([]() -> const OpKernel* { return NewOpKernel<T>(); });
  }
  template<typename T>
  OpKernelRegistry& SetIsMatchedHob(const T& hob) {
    result_.is_matched_hob = std::make_shared<T>(hob);
    return *this;
  }
  OpKernelRegistry& SetInferTmpSizeFn(InferTmpSizeFn fn);
  OpKernelRegistry& SetInplaceProposalFn(InplaceProposalFn fn);

  Maybe<OpKernelRegistry&> Finish();
  OpKernelRegistryResult GetResult() { return result_; }

  OpKernelRegistry& SetCreateFn(OpKernelCreateFn fn);
  OpKernelRegistry& SetPriority(int32_t priority);

 private:
  OpKernelRegistryResult result_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_KERNEL_REGISTRY_H_
