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
#include "fmt/core.h"
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
#include "oneflow/core/functional/impl/common.h"

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
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dim");

    const auto do_dispatch_base_on_device = [&attrs, this](
                                                const std::shared_ptr<one::Tensor>& input,
                                                const std::shared_ptr<one::Tensor>& other,
                                                const int64_t dim) -> Maybe<Tensor> {
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

    Shape shape_to_broadcast;
    std::deque<bool> need_to_broadcast;

    std::tie(shape_to_broadcast, need_to_broadcast) =
        *JUST(InferUnifiedShapeForBroadcastingWithInfo({*input->shape(), *other->shape()}));
    CHECK_EQ_OR_RETURN(need_to_broadcast.size(), 2)
        << fmt::format("The number of boolean values to determine if the tensor is to be broadcast "
                       "should be 2 (which is {})",
                       need_to_broadcast.size());
    const auto new_input =
        need_to_broadcast[0] ? JUST(functional::Expand(input, shape_to_broadcast)) : input;
    const auto new_other =
        need_to_broadcast[1] ? JUST(functional::Expand(other, shape_to_broadcast)) : other;

    if (!dim.has_value()) {
      return do_dispatch_base_on_device(new_input, new_other,
                                        JUST(FindValidDim(shape_to_broadcast)));
    }

    int64_t new_dim = JUST(dim);
    if (new_dim < 0) { new_dim += shape_to_broadcast.NumAxes(); }
    CHECK_EQ_OR_RETURN(shape_to_broadcast.At(new_dim), 3)
        << Error::RuntimeError()
        << fmt::format("the size of the specified dimension(which is {}) is not 3.", JUST(dim));

    return do_dispatch_base_on_device(new_input, new_other, new_dim);
  }

 private:
  Maybe<int64_t> FindValidDim(const Shape& shape) const {
    int64_t valid_dim = -1;
    const auto& dim_vec = shape.dim_vec();
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