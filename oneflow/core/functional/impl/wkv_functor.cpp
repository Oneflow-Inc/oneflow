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
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class WkvForwardFunctor {
 public:
  WkvForwardFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("wkv_forward")
                         .Input("w")
                         .Input("u")
                         .Input("k")
                         .Input("v")
                         .Input("y")
                         .Output("y_")
                         .Build());
  }
  Maybe<void> operator()(const int64_t B, const int64_t T, const int64_t C,
                         const std::shared_ptr<one::Tensor>& w,
                         const std::shared_ptr<one::Tensor>& u,
                         const std::shared_ptr<one::Tensor>& k,
                         const std::shared_ptr<one::Tensor>& v,
                         const std::shared_ptr<one::Tensor>& y) const {
    MutableAttrMap attrs;
    attrs.SetAttr<int64_t>("B", B);
    attrs.SetAttr<int64_t>("T", T);
    attrs.SetAttr<int64_t>("C", C);
    OpExprInterpContext ctx(attrs);
    OpInterpUtil::Dispatch<Tensor>(*op_, {w, u, k, v, y}, ctx);
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class WkvBackwardFunctor {
 public:
  WkvBackwardFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("wkv_backward")
                         .Input("w")
                         .Input("u")
                         .Input("k")
                         .Input("v")
                         .Input("gy")
                         .Input("gw")
                         .Input("gu")
                         .Input("gk")
                         .Input("gv")
                         .Output("y")
                         .Build());
  }
  Maybe<void> operator()(
      const int64_t B, const int64_t T, const int64_t C, const std::shared_ptr<one::Tensor>& w,
      const std::shared_ptr<one::Tensor>& u, const std::shared_ptr<one::Tensor>& k,
      const std::shared_ptr<one::Tensor>& v, const std::shared_ptr<one::Tensor>& gy,
      const std::shared_ptr<one::Tensor>& gw, const std::shared_ptr<one::Tensor>& gu,
      const std::shared_ptr<one::Tensor>& gk, const std::shared_ptr<one::Tensor>& gv) const {
    MutableAttrMap attrs;
    attrs.SetAttr<int64_t>("B", B);
    attrs.SetAttr<int64_t>("T", T);
    attrs.SetAttr<int64_t>("C", C);
    OpExprInterpContext ctx(attrs);
    OpInterpUtil::Dispatch<Tensor>(*op_, {w, u, k, v, gy, gw, gu, gk, gv}, ctx);
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

using namespace impl;

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<WkvForwardFunctor>("WkvForward");
  m.add_functor<WkvBackwardFunctor>("WkvBackward");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
