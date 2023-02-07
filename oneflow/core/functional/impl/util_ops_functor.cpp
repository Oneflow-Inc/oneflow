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

#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/impl/common.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class UtilOpsFunctor {
 public:
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& input) const {
    return JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {input}));
  }

 protected:
  std::shared_ptr<OpExpr> op_;
};

class IsNanFunctor final : public UtilOpsFunctor {
 public:
  IsNanFunctor() { op_ = CHECK_JUST(one::OpBuilder("isnan").Input("in").Output("out").Build()); }
};

class IsInfFunctor final : public UtilOpsFunctor {
 public:
  IsInfFunctor() { op_ = CHECK_JUST(one::OpBuilder("isinf").Input("in").Output("out").Build()); }
};

class IsFiniteFunctor final : public UtilOpsFunctor {
 public:
  IsFiniteFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("isfinite").Input("in").Output("out").Build());
  }
};

class DependFunctor {
 public:
  DependFunctor() {
    op_ = CHECK_JUST(
        one::OpBuilder("depend").Input("in").Input("depend_tensor").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const std::shared_ptr<one::Tensor>& depend_tensor) const {
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in, depend_tensor});
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class DependTupleFunctor {
 public:
  DependTupleFunctor() {
    ops_.resize(kMaxInputCount);
    for (int n = 0; n < ops_.size(); ++n) {
      ops_[n] = CHECK_JUST(
          one::OpBuilder("depend").Input("in").Input("depend_tensor").Output("out").Build());
    }
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& in,
                           const one::TensorTuple& depends) const {
    return _dispatch(in, depends, 0);
  }

 private:
  Maybe<Tensor> _dispatch(const std::shared_ptr<one::Tensor>& in, const one::TensorTuple& depends,
                          const int pos) const {
    const size_t ndepend = depends.size();
    Maybe<Tensor> output = OpInterpUtil::Dispatch<Tensor>(*ops_[pos], {in, depends[pos]});
    if (pos == ndepend - 1) { return output; }
    return _dispatch(JUST(output), depends, pos + 1);
  }

  std::vector<std::shared_ptr<OpExpr>> ops_;
};

}  // namespace impl

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<IsNanFunctor>("IsNan"); };
ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<IsInfFunctor>("IsInf"); };
ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<IsFiniteFunctor>("IsFinite"); };
ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<DependFunctor>("Depend"); };
ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<DependTupleFunctor>("DependTuple"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
