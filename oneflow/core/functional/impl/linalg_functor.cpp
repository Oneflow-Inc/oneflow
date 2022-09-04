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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/error.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/tensor_desc.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {
namespace linalg {

class CrossFunctor {
 public:
  CrossFunctor() {
    op_ = CHECK_JUST(OpBuilder("linalg_cross").Input("input").Input("other").Output("out").Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& input,
                           const std::shared_ptr<one::Tensor>& other,
                           const Optional<int64_t>& dim) const {
    CHECK_EQ_OR_RETURN(*input->shape(), *other->shape())
        << Error::RuntimeError() << "input and other should have same shape.";

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim");

    const auto do_dispatch_base_on_device = [&](const int64_t dim) -> Maybe<Tensor> {
      DeviceType device{};

      if (input->is_global()) {
        device = JUST(input->parallel_desc())->device_type();
      } else {
        device = JUST(input->device())->enum_type();
      }

      const int64_t final_dim = input->ndim() - 1;

      if (device == DeviceType::kCUDA && dim != final_dim) {
        attrs.SetAllAttrs(final_dim);

        std::vector<int> perm(input->ndim(), 0);
        for (size_t i = 0; i < perm.size(); ++i) { perm[i] = static_cast<int>(i); }
        std::swap(perm[dim], perm[final_dim]);
        return functional::Transpose(
            JUST(OpInterpUtil::Dispatch<Tensor>(*op_,
                                                {JUST(functional::Transpose(input, perm)),
                                                 JUST(functional::Transpose(other, perm))},
                                                attrs)),
            perm);
      }

      attrs.SetAllAttrs(dim);
      return OpInterpUtil::Dispatch<Tensor>(*op_, {input, other}, attrs);
    };

    if (!dim.has_value()) { return do_dispatch_base_on_device(JUST(FindValidDim(input))); }

    int64_t new_dim = JUST(dim);
    if (new_dim < 0) { new_dim += input->ndim(); }

    return do_dispatch_base_on_device(new_dim);
  }

 private:
  Maybe<int64_t> FindValidDim(const std::shared_ptr<one::Tensor>& t) const {
    int64_t valid_dim = -1;
    const auto& dim_vec = t->shape()->dim_vec();
    for (size_t i = 0; i < dim_vec.size(); ++i) {
      if (dim_vec[i] == 3) {
        valid_dim = i;
        break;
      }
    }
    if (valid_dim == -1) { return Error::RuntimeError() << "no dimension of size 3 in input."; }
    return valid_dim;
  }

  std::shared_ptr<OpExpr> op_;
};

}  // namespace linalg
}  // namespace impl

using namespace impl::linalg;

ONEFLOW_FUNCTION_LIBRARY(m) { m.add_functor<CrossFunctor>("LinalgCross"); }

}  // namespace functional
}  // namespace one
}  // namespace oneflow