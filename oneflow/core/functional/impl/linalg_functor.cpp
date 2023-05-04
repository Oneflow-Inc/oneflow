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

class MultiDotFunctor {
 public:
  Maybe<Tensor> operator()(const TensorTuple& tensors) const {
    const size_t n = tensors.size();
    CHECK_GE_OR_RETURN(n, 2) << Error::RuntimeError()
                             << "multi_dot(): expected at least 2 tensors but got " << n;
    CHECK_LT_OR_RETURN(n, kMaxInputCount);

    std::vector<int64_t> out_shape;
    TensorTuple tensors_for_calculate = tensors;

    // If the first tensor is 1D of size n view it as a row vector (1, n)
    if (tensors[0]->ndim() == 1) {
      tensors_for_calculate[0] = JUST(functional::Unsqueeze(tensors[0], 0));
    } else if (tensors[0]->ndim() == 2) {
      tensors_for_calculate[0] = tensors[0];
      out_shape.emplace_back(tensors_for_calculate[0]->shape()->At(0));
    } else {
      CHECK_LE_OR_RETURN(tensors[0]->ndim(), 2)
          << Error::RuntimeError() << "multi_dot(): the first tensor must be 1D or 2D but got "
          << tensors[0]->ndim();
    }

    // If the last tensor is 1D of size n view it as a column vector (n, 1)
    if (tensors[n - 1]->ndim() == 1) {
      tensors_for_calculate[n - 1] =
          JUST(functional::Unsqueeze(tensors[n - 1], tensors[n - 1]->ndim()));
    } else if (tensors[n - 1]->ndim() == 2) {
      tensors_for_calculate[n - 1] = tensors[n - 1];
      out_shape.emplace_back(tensors_for_calculate[n - 1]->shape()->At(1));
    } else {
      CHECK_LE_OR_RETURN(tensors[n - 1]->ndim(), 2)
          << Error::RuntimeError() << "multi_dot(): the last tensor must be 1D or 2D but got "
          << tensors[n - 1]->ndim();
    }

    // Ensure middle tensors are 2D
    const auto dtype = tensors_for_calculate[0]->dtype();
    const auto device = JUST(tensors_for_calculate[0]->device());
    CHECK_LE_OR_RETURN(tensors_for_calculate[0]->ndim(), 2)
        << Error::RuntimeError() << "multi_dot(): tensors' dim should be lower equal than 2.";
    for (int64_t i = 1; i < tensors_for_calculate.size(); ++i) {
      CHECK_LE_OR_RETURN(tensors_for_calculate[i]->ndim(), 2)
          << Error::RuntimeError()
          << "multi_dot(): tensors' dim size should be lower equal than 2.";
      CHECK_OR_RETURN(tensors_for_calculate[i]->dtype() == dtype)
          << Error::RuntimeError()
          << "multi_dot(): all tensors must have be the same dtype but tensor 0 is "
          << dtype->name() << "and tensor" << i << " is "
          << tensors_for_calculate[i]->dtype()->name();
      CHECK_OR_RETURN(JUST(tensors_for_calculate[i]->device()) == device)
          << Error::RuntimeError()
          << "multi_dot(): all tensors must have be the same device but tensor 0 is "
          << device->ToString() << "and tensor" << i << " is "
          << JUST(tensors_for_calculate[i]->device())->ToString();
    }

    // for 2 matrices
    if (tensors_for_calculate.size() == 2) {
      std::shared_ptr<Tensor> result =
          JUST(functional::MatMul(tensors_for_calculate[0], tensors_for_calculate[1],
                                  /*transpose_a=*/false, /*transpose_b=*/false, /*alpha=*/1));
      return view::Reshape(result, Shape(out_shape));
    }

    // for 3 matrices
    if (tensors_for_calculate.size() == 3) {
      const int64_t a = tensors_for_calculate[0]->shape()->At(0);
      const int64_t b = tensors_for_calculate[1]->shape()->At(0);
      const int64_t c = tensors_for_calculate[2]->shape()->At(0);
      const int64_t d = tensors_for_calculate[2]->shape()->At(1);

      // The matrices are of size (a x b), (b x c), (c x d)
      // cost_1 is the cost of parenthesizing (a x b) and (b x c) and then
      // combining (c x d) cost_2 is the cost of parenthesizing (b x c) and (c x
      // d) and then combining (a x b)
      const int64_t cost_1 = (a * c) * (b + d);
      const int64_t cost_2 = (b * d) * (a + c);

      if (cost_1 > cost_2) {
        std::shared_ptr<Tensor> result =
            JUST(functional::MatMul(tensors_for_calculate[1], tensors_for_calculate[2],
                                    /*transpose_a=*/false, /*transpose_b=*/false, /*alpha=*/1));
        result = JUST(functional::MatMul(tensors_for_calculate[0], result, /*transpose_a=*/false,
                                         /*transpose_b=*/false, /*alpha=*/1));
        return view::Reshape(result, Shape(out_shape));
      } else {
        std::shared_ptr<Tensor> result =
            JUST(functional::MatMul(tensors_for_calculate[0], tensors_for_calculate[1],
                                    /*transpose_a=*/false, /*transpose_b=*/false, /*alpha=*/1));
        result = JUST(functional::MatMul(result, tensors_for_calculate[2], /*transpose_a=*/false,
                                         /*transpose_b=*/false, /*alpha=*/1));
        return view::Reshape(result, Shape(out_shape));
      }
    }

    // for 4 or more matrices
    auto matrix_chain_multiplication = [](TensorTuple tensors,
                                          const std::vector<std::vector<int64_t>>& order, int64_t i,
                                          int64_t j) -> Maybe<Tensor> {
      if (i == j) { return tensors[i]; }
      std::function<Maybe<Tensor>(TensorTuple, const std::vector<std::vector<int64_t>>&, int64_t,
                                  int64_t)>
          wrap_matrix_chain_multiplication;
      return functional::MatMul(
          JUST(wrap_matrix_chain_multiplication(tensors, order, i, order[i][j])),
          JUST(wrap_matrix_chain_multiplication(tensors, order, order[i][j] + 1, j)),
          /*transpose_a=*/false, /*transpose_b=*/false, /*alpha=*/1);
    };

    const auto order = matrix_chain_order(tensors_for_calculate);
    const int64_t i = 0;
    const int64_t j = n - 1;
    std::shared_ptr<Tensor> result =
        JUST(matrix_chain_multiplication(tensors_for_calculate, order, i, j));
    return view::Reshape(result, Shape(out_shape));
  }

 private:
  std::vector<std::vector<int64_t>> matrix_chain_order(TensorTuple tensors) const {
    const size_t n = tensors.size();

    // Tensor i has dimensions p[i] x p[i + 1]
    std::vector<int64_t> p(n + 1);
    for (int64_t i = 0; i < n; i++) { p[i] = tensors[i]->shape()->At(0); }
    p[n] = tensors[n - 1]->shape()->At(1);

    // m[i, j] = k where k is the minimum cost for multiplying tensors i...j
    std::vector<std::vector<int64_t>> m(n, std::vector<int64_t>(n, 0));

    // s[i, j] = k where k is the index at which to split the list such that
    // optimally multiplying matrices i...k and k...j first and then the resulting
    // matrices is the optimal order for multiplying matrices i...j.
    std::vector<std::vector<int64_t>> s(n, std::vector<int64_t>(n));

    // Compute the optimal multiplication order
    for (int64_t l = 1; l < n; l++) {
      for (int64_t i = 0; i < n - l; i++) {
        const auto j = i + l;
        m[i][j] = std::numeric_limits<int64_t>::max();
        for (int64_t k = i; k < j; k++) {
          const auto q = m[i][k] + m[k + 1][j] + p[i] * p[k + 1] * p[j + 1];
          if (q < m[i][j]) {
            m[i][j] = q;
            s[i][j] = k;
          }
        }
      }
    }

    return s;
  }
};

}  // namespace linalg
}  // namespace impl

using namespace impl::linalg;

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<CrossFunctor>("LinalgCross");
  m.add_functor<MultiDotFunctor>("MultiDot");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow