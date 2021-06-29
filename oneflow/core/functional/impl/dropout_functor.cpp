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

#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/user/kernels/random_mask_like_kernel.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {
class DropoutFunctor {
 public:
  DropoutFunctor() {
    random_mask_like_op_ =
        CHECK_JUST(one::OpBuilder("random_mask_like").Input("like").Output("out").Build());
    dropout_op_ =
        CHECK_JUST(one::OpBuilder("dropout").Input("in").Input("mask").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& p,
                           const Optional<one::Generator>& generator) const {
    MutableAttrMap random_mask_like_attrs;
    JUST(random_mask_like_attrs.SetAttr<float>("rate", p));

    std::shared_ptr<one::Generator> gen;
    if (!generator) {
      gen = one::Generator::GetDefaultGenerator();
    } else {
      gen = JUST(generator.value());
    }
  
    JUST(random_mask_like_attrs.SetAttr<int64_t>("seed", gen->get_current_seed()));
    const auto& random_mask_like_state = std::make_shared<RandomMaskLikeKernelState>(gen);

    const auto& mask = JUST(OpInterpUtil::Dispatch<Tensor>(
        *random_mask_like_op_, {x}, {random_mask_like_attrs, random_mask_like_state}));
    float scale = 1.0;
    if (p != 1.0) { scale = 1.0 / (1.0 - p); }
    MutableAttrMap dropout_attrs;
    JUST(dropout_attrs.SetAttr<float>("scale", scale));
    return OpInterpUtil::Dispatch<Tensor>(*dropout_op_, {x, mask}, dropout_attrs);
  }

 private:
  std::shared_ptr<OpExpr> random_mask_like_op_;
  std::shared_ptr<OpExpr> dropout_op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<impl::DropoutFunctor>("Dropout"); };

}  // namespace functional
}  // namespace one
}  // namespace oneflow
