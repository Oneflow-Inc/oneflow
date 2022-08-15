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
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/impl/common.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class GradAccRepeatFunctor {
 public:
  GradAccRepeatFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("repeat").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& in, int32_t repeat_num) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("repeat_num", repeat_num));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GradAccCollectFunctor {
 public:
  GradAccCollectFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("acc").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& in, int32_t collect_num) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("max_acc_num", collect_num));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GradAccPackFunctor {
 public:
  GradAccPackFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("pack").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& in, int32_t pack_num) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("pack_num", pack_num));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class GradAccUnpackFunctor {
 public:
  GradAccUnpackFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("unpack").Input("in").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<Tensor>& in, int32_t unpack_num) const {
    MutableAttrMap attrs;
    JUST(attrs.SetAttr<int32_t>("unpack_num", unpack_num));
    return OpInterpUtil::Dispatch<Tensor>(*op_, {in}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::GradAccRepeatFunctor>("GradAccRepeat");
  m.add_functor<impl::GradAccCollectFunctor>("GradAccCollect");
  m.add_functor<impl::GradAccPackFunctor>("GradAccPack");
  m.add_functor<impl::GradAccUnpackFunctor>("GradAccUnpack");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
