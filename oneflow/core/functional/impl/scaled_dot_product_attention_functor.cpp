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

#include <cmath>
#include <cstdint>
#include <memory>
#include "fmt/core.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/scalar.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/user/kernels/distributions/common.h"
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/user/kernels/scaled_dot_product_attention_kernel.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

template<int alignment_size>
Maybe<one::Tensor> pad_last_dim(const std::shared_ptr<one::Tensor>& input) {
  auto num_dims = input->shape()->NumAxes();
  auto last_dim_size = input->shape()->At(num_dims - 1);
  if (last_dim_size % alignment_size == 0) { return input; }
  auto pad_count = alignment_size - (last_dim_size % alignment_size);

  return JUST(functional::Pad(input, {0, pad_count}, "constant", Scalar(0)));
  ;
}

}  // namespace

class ScaledDotProductFlashAttentionFunctor {
 public:
  ScaledDotProductFlashAttentionFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("scaled_dot_product_flash_attention")
                         .Input("query")
                         .Input("key")
                         .Input("value")
                         .Output("out")
                         .Output("softmax_lse")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& query,
                           const std::shared_ptr<one::Tensor>& key,
                           const std::shared_ptr<one::Tensor>& value,
                           const Optional<one::Tensor>& attn_mask, const float& dropout_p,
                           const bool& is_causal, const Optional<float>& scale,
                           const int64_t& seed = 0) const {
    const auto og_size = query->shape()->At(3);
    const auto batch_size = query->shape()->At(0);
    const auto seqlen_q = query->shape()->At(2);
    const auto num_heads = query->shape()->At(1);
    // const auto max_seqlen_batch_q = query->shape()->At(2);
    const auto max_seqlen_batch_k = key->shape()->At(2);
    const auto max_seqlen_batch_v = value->shape()->At(2);

    CHECK_EQ(max_seqlen_batch_k, max_seqlen_batch_v);

    // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
    // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
    // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
    auto q_padded = JUST(pad_last_dim<8>(query));
    auto k_padded = JUST(pad_last_dim<8>(key));
    auto v_padded = JUST(pad_last_dim<8>(value));

    auto q_ = JUST(functional::Transpose(q_padded, {0, 2, 1, 3}));
    auto k_ = JUST(functional::Transpose(k_padded, {0, 2, 1, 3}));
    auto v_ = JUST(functional::Transpose(v_padded, {0, 2, 1, 3}));
    // Query -> Query(Batch x Q_seq_len  x Num_heads x Dim_per_head)
    // Key   -> Key  (Batch x KV_seq_len x Num_heads x Dim_per_head)
    // Value -> Value(Batch x KV_seq_len x Num_heads x Dim_per_head)

    const auto& scale_ =
        scale.has_value() ? scale : (1.0f / std::sqrt(static_cast<float>(query->shape()->At(3))));

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("p_dropout", "softmax_scale", "is_causal",
                                                 "window_size_left", "window_size_right", "seed");
    attrs.SetAllAttrs(dropout_p, scale_, is_causal, -1, -1, seed);

    auto gen = JUST(one::DefaultAutoGenerator());
    gen = JUST(GetGeneratorForLazyOrGlobal(gen, LazyMode::is_enabled(), query));
    const auto& state = std::make_shared<ScaledDotProductFlashAttentionKernelState>(gen);
    OpExprInterpContext ctx(attrs, state);

    std::shared_ptr<one::Tensor> output_ =
        JUST(OpInterpUtil::Dispatch<one::Tensor>(*op_, {q_, k_, v_}, ctx));

    auto output_padded = JUST(functional::Transpose(output_, {0, 2, 1, 3}));
    return JUST(functional::Slice(output_padded, {0, 0, 0, 0}, {batch_size, num_heads, seqlen_q, og_size}, {1, 1, 1, 1}, false));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::ScaledDotProductFlashAttentionFunctor>("ScaledDotProductFlashAttention");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
