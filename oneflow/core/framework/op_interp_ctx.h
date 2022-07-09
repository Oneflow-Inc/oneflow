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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERP_CTX_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERP_CTX_H_

#include <string>

#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_attrs.h"
#include "oneflow/core/framework/op_base.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.h"

namespace oneflow {

using user_op::AttrVal;
template<typename T>
using TypedAttrValRef = user_op::TypedAttrValRef<T>;

namespace user_op {
class OpKernelState;
}  // namespace user_op

class OpInterpCtx {
 public:
  explicit OpInterpCtx(const std::shared_ptr<OpBase>& op) : op_(op) {}
  virtual ~OpInterpCtx() = default;

  template<typename T>
  Maybe<const T&> GetAttr(const std::string& attr_name) const;

  Maybe<AttrVal> GetAttr(const std::string& attr_name) const;

  OpAttrs GetAttrs() const;

  template<typename T>
  Maybe<void> SetAttr(const std::string& attr_name, const T& attr_val);

  Maybe<void> SetAttr(const std::string& attr_name, const AttrVal& attr_val);

  bool HasAttr(const std::string& attr_name) const;

  const HashSet<std::string>& AttrNames() const;

 public:
  std::shared_ptr<OpBase> op_;

  Optional<Symbol<Device>> device;               // for local op
  Optional<Symbol<ParallelDesc>> parallel_desc;  // for global op
  Optional<Symbol<NdSbp>> sbp;                   // for global op
  Optional<user_op::OpKernelState> state;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERP_CTX_H_
