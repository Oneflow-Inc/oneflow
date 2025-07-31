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

#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/error.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/functional/impl/binary_functor.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/autograd/autograd_mode.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class ScaledDotProductAttentionMathFunctor {
 public:
  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& query,
                                const std::shared_ptr<one::Tensor>& key,
                                const std::shared_ptr<one::Tensor>& value,
                                const Optional<one::Tensor>& attn_mask, float dropout_p,
                                bool is_causal, const Optional<one::Tensor>& dropout_mask) const {
    // Naive, composite implementation defined here.

    // Scale q,k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    int64_t ndim_idx = query->ndim();
    double embed_size = query->shape()->At(ndim_idx - 1);

    auto scaling_factor = std::sqrt(std::sqrt(embed_size));

    // const auto query = query_ / scaling_factor;
    const auto query_ = JUST(functional::ScalarDiv(query, Scalar(scaling_factor)));

    std::shared_ptr<Tensor> attn_mask_t = JUST(attn_mask);

    if (is_causal) {
      // CHECK_OR_RETURN(!attn_mask_->has_value()) << "_scaled_dot_product_attention: Explicit
      // attn_mask should not be set when is_causal=True"; CHECK_OR_RETURN(!query_->is_consistent()
      // && !key->is_consistent()) << "_scaled_dot_product_attention: Nested tensors for query / key
      // are not supported when is_causal=True"; Replace attn_mask with causal mask; lower
      // triangular elements take part in attention.
      const auto L = query_->shape()->At(ndim_idx - 2), S = key->shape()->At(ndim_idx - 2);

      auto empty = JUST(functional::Empty({L, S}, query_->dtype(), JUST(query_->device()),
                                          query_->requires_grad(),
                                          /*pin_memory=*/false));
      empty = JUST(functional::Fill(empty, Scalar(1.0)));
      attn_mask_t = JUST(functional::Triu(empty, 0.0));
    }

    if (attn_mask.has_value()) {
      // Convert boolean mask to additive mask; need to invert mask to indicate what to mask *out*.
      if (attn_mask_t->dtype() == DType::Bool()) {
        auto new_attn_mask = JUST(functional::Empty(*(attn_mask_t->shape()), query_->dtype(),
                                                    JUST(query_->device()), query_->requires_grad(),
                                                    /*pin_memory=*/false));
        auto mask = JUST(functional::LogicalNot(attn_mask_t));
        attn_mask_t = JUST(functional::MaskedFill(
            new_attn_mask, mask, Scalar(-std::numeric_limits<double>::infinity())));
      }
      // Otherwise, attn_mask represents an additive attention tensor
    }

    auto key_ = JUST(functional::Transpose2dim(key, -2, -1));
    key_ = JUST(functional::ScalarDiv(key_, Scalar(scaling_factor)));

    auto attn = JUST(functional::MatMul(query_, key_, false, false, 1.0));

    if (attn_mask.has_value()) {
      JUST(functional::Add(attn, attn_mask_t, /*alpha*/ Scalar(1), /*inplace*/ true));
    }

    attn = JUST(functional::Softmax(attn, -1));

    if (dropout_p > 0.0) {
      if (dropout_mask.has_value()) {
        std::shared_ptr<Tensor> temp_tensor = JUST(dropout_mask);
        temp_tensor = JUST(functional::LogicalNot(temp_tensor));
        auto attn_dropout_masked = JUST(functional::MaskedFill(attn, temp_tensor, Scalar(0)));
        auto dropout_scaling = 1.0 / (1 - dropout_p);
        auto dropout_vaule =
            JUST(functional::ScalarMul(value, Scalar(dropout_scaling), /*inplace*/ false));
        auto out = JUST(functional::MatMul(attn_dropout_masked, dropout_vaule, false, false, 1.0));
        return one::TensorTuple({out, attn});
      } else {
        auto generator = JUST(one::DefaultAutoGenerator());
        attn = JUST(functional::Dropout(attn, dropout_p, true, false, generator, nullptr));
      }
    }
    auto out = JUST(functional::MatMul(attn, value, false, false, 1.0));
    return one::TensorTuple({out, attn});
  }
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ScaledDotProductAttentionMathFunctor>("ScaledDotProductAttentionMath");
};

}  // namespace functional
}  // namespace one
}  // namespace oneflow
