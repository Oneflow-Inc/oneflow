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
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/functional/function_library.h"
#include "oneflow/core/functional/sequence_function.h"
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/functional/impl/unary_functor.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/user/kernels/random_mask_like_kernel.h"
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/user/kernels/distributions/common.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

namespace {

Maybe<void> ParseDims(const std::string& name, const Shape& shape, const std::string& layout,
                      const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
                      int64_t* b, int64_t* m, int64_t* h, int64_t* k) {
  if (shape.NumAxes() == 3) {
    if (layout == "BM(HK)" || layout == "MB(HK)" || layout == "BM(H2K)" || layout == "MB(H2K)"
        || layout == "BM(H3K)" || layout == "MB(H3K)") {
      int64_t packed_n = 0;
      if (layout == "BM(HK)") {
        *b = shape.At(0);
        *m = shape.At(1);
        packed_n = 1;
      } else if (layout == "MB(HK)") {
        *b = shape.At(1);
        *m = shape.At(0);
        packed_n = 1;
      } else if (layout == "BM(H2K)") {
        CHECK_NE_OR_RETURN(name, "query") << "query_layout should not be 'BM(H2K)'";
        *b = shape.At(0);
        *m = shape.At(1);
        packed_n = 2;
      } else if (layout == "MB(H2K)") {
        CHECK_NE_OR_RETURN(name, "query") << "query_layout should not be 'MB(H2K)'";
        *b = shape.At(1);
        *m = shape.At(0);
        packed_n = 2;
      } else if (layout == "BM(H3K)") {
        *b = shape.At(0);
        *m = shape.At(1);
        packed_n = 3;
      } else if (layout == "MB(H3K)") {
        *b = shape.At(1);
        *m = shape.At(0);
        packed_n = 3;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
      const int64_t hidden_size = shape.At(2);
      if (num_heads) {
        const int64_t expected_h = JUST(num_heads);
        const int64_t packed_h = packed_n * expected_h;
        CHECK_EQ_OR_RETURN(hidden_size % packed_h, 0)
            << "The size of the last dimension of the " << name
            << " tensor should be a multiple of " << packed_h << ".";
        *h = expected_h;
        *k = hidden_size / packed_h;
      } else if (head_size) {
        const int64_t expected_k = JUST(head_size);
        const int64_t packed_k = expected_k * packed_n;
        CHECK_EQ_OR_RETURN(hidden_size % packed_k, 0)
            << "The size of the last dimension of the " << name
            << " tensor should be a multiple of " << packed_k << ".";
        *h = hidden_size / packed_k;
        *k = expected_k;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << name
          << "_layout should be 'BM(HK)', 'MB(HK)', 'BM(H2K)', 'MB(H2K)', 'BM(H3K)' or "
             "'MB(H3K)' when the number of dimensions of "
          << name << " tensor is 3.";
    }
  } else if (shape.NumAxes() == 4) {
    if (layout == "BMHK") {
      *b = shape.At(0);
      *m = shape.At(1);
      *h = shape.At(2);
      *k = shape.At(3);
    } else if (layout == "BHMK") {
      *b = shape.At(0);
      *m = shape.At(2);
      *h = shape.At(1);
      *k = shape.At(3);
    } else if (layout == "MBHK") {
      *b = shape.At(1);
      *m = shape.At(0);
      *h = shape.At(2);
      *k = shape.At(3);
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << name << "_layout should be 'BMHK', 'BHMK' or 'MBHK' when the number of dimensions of "
          << name << " tensor is 4.";
    }
    if (num_heads) {
      const int64_t expected_h = JUST(num_heads);
      CHECK_EQ_OR_RETURN(*h, expected_h)
          << "The size of dimension 'H' of " << name << " tensor should be " << expected_h << ".";
    }
    if (head_size) {
      const int64_t expected_k = JUST(head_size);
      CHECK_EQ_OR_RETURN(*k, expected_k)
          << "The size of dimension 'K' of " << name << " tensor should be " << expected_k << ".";
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "The number of dimensions of the " << name
                                << " tensor should be 3 or 4";
  };
  return Maybe<void>::Ok();
}

}  // namespace

class FusedMultiHeadAttentionInferenceFunctor {
 public:
  FusedMultiHeadAttentionInferenceFunctor() = default;
  Maybe<Tensor> operator()(
      const std::shared_ptr<one::Tensor>& query, const std::shared_ptr<one::Tensor>& key,
      const std::shared_ptr<one::Tensor>& value, const int64_t& num_heads, const bool& causal,
      const int64_t& query_hidden_slice_start, const int64_t& query_hidden_slice_end,
      const int64_t& key_hidden_slice_start, const int64_t& key_hidden_slice_end,
      const int64_t& value_hidden_slice_start, const int64_t& value_hidden_slice_end,
      const Optional<one::Tensor>& attn_bias, const int64_t& causal_diagonal_offset) const {
    CHECK_OR_RETURN(query_hidden_slice_start == 0 && key_hidden_slice_start == 0
                    && value_hidden_slice_start == 0 && query_hidden_slice_end == -1
                    && key_hidden_slice_end == -1 && value_hidden_slice_end == -1)
        << "The parameters 'query_hidden_slice_start', 'query_hidden_slice_end', "
           "'key_hidden_slice_start', 'key_hidden_slice_end', 'value_hidden_slice_start', "
           "'value_hidden_slice_end' have been deprecated.";

    const int64_t query_hidden_size = query->shape()->At(2);
    CHECK_EQ_OR_RETURN(query_hidden_size % num_heads, 0)
        << "The hidden size of the query tensor should be a multiple of num_heads.";
    const int64_t query_head_size = query_hidden_size / num_heads;
    return functional::FusedMultiHeadAttentionInferenceV2(query, "BM(HK)", query_head_size, key,
                                                          "BM(HK)", value, "BM(HK)", attn_bias,
                                                          "BM(HK)", causal, causal_diagonal_offset);
  }
};

class FusedMultiHeadAttentionInferenceV2Functor {
 public:
  FusedMultiHeadAttentionInferenceV2Functor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_multi_head_attention_inference")
                         .Input("query")
                         .Input("key")
                         .Input("value")
                         .Output("out")
                         .Build());
    op_with_attn_bias_ = CHECK_JUST(one::OpBuilder("fused_multi_head_attention_inference")
                                        .Input("query")
                                        .Input("key")
                                        .Input("value")
                                        .Input("attn_bias")
                                        .Output("out")
                                        .Build());
  }
  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& query,
                           const std::string& query_layout,
                           const Optional<int64_t>& query_head_size,
                           const Optional<one::Tensor>& key,
                           const Optional<std::string>& key_layout,
                           const Optional<one::Tensor>& value,
                           const Optional<std::string>& value_layout,
                           const Optional<one::Tensor>& attn_bias, const std::string& output_layout,
                           const bool& causal, const int64_t& causal_diagonal_offset) const {
    CHECK_GE_OR_RETURN(causal_diagonal_offset, 0)
        << "The value of causal_diagonal_offset should be greater or equal to 0.";

    std::shared_ptr<one::Tensor> key_tensor;
    std::string key_tensor_layout;
    std::shared_ptr<one::Tensor> value_tensor;
    std::string value_tensor_layout;

    int64_t q_b = 0;
    int64_t q_m = 0;
    int64_t q_h = 0;
    int64_t q_k = 0;
    JUST(ParseDims("query", *query->shape(), query_layout, Optional<int64_t>(), query_head_size,
                   &q_b, &q_m, &q_h, &q_k));
    CHECK_EQ_OR_RETURN(q_k % 8, 0)
        << "The size of dimension 'K' of the query tensor should be a multiple of 8.";

    int64_t k_b = 0;
    int64_t k_m = 0;
    int64_t k_h = 0;
    int64_t k_k = 0;
    if (key) {
      key_tensor = JUST(key);
      key_tensor_layout = *JUST(key_layout);
      JUST(ParseDims("key", *key_tensor->shape(), key_tensor_layout, Optional<int64_t>(), q_k, &k_b,
                     &k_m, &k_h, &k_k));
      CHECK_EQ_OR_RETURN(k_b, q_b) << "The size of dimension 'B' of the key tensor should be the "
                                      "same as that of the query tensor.";
      CHECK_EQ_OR_RETURN(k_h, q_h) << "The size of dimension 'H' of the key tensor should be the "
                                      "same as that of the query tensor.";

    } else {
      CHECK_OR_RETURN(query_layout == "BM(H3K)" || query_layout == "MB(H3K)")
          << "The value of query_layout should be 'BM(H3K)' or 'MB(H3K)' when the key tensor is "
             "None.";
      key_tensor = query;
      key_tensor_layout = query_layout;
      k_b = q_b;
      k_m = q_m;
      k_h = q_h;
      k_k = q_k;
    }

    int64_t v_b = 0;
    int64_t v_m = 0;
    int64_t v_h = 0;
    int64_t v_k = 0;
    if (value) {
      value_tensor = JUST(value);
      value_tensor_layout = *JUST(value_layout);
      JUST(ParseDims("value", *value_tensor->shape(), value_tensor_layout, q_h, Optional<int64_t>(),
                     &v_b, &v_m, &v_h, &v_k));
      CHECK_EQ_OR_RETURN(v_b, q_b) << "The size of dimension 'B' of the value tensor should be the "
                                      "same as that of the query tensor.";
      CHECK_EQ_OR_RETURN(v_m, k_m) << "The size of dimension 'M' of the value tensor should be the "
                                      "same as that of the key tensor.";
      CHECK_EQ_OR_RETURN(v_k % 8, 0)
          << "The size of dimension 'K' of the value tensor should be a multiple of 8.";

    } else {
      CHECK_OR_RETURN(key_tensor_layout == "BM(H2K)" || key_tensor_layout == "MB(H2K)"
                      || key_tensor_layout == "BM(H3K)" || key_tensor_layout == "MB(H3K)")
          << "The value of key_layout should be 'BM(H3K)', 'MB(H3K)', 'BM(H2K)' or 'MB(H2K)' when "
             "the value tensor is None.";
      value_tensor = key_tensor;
      value_tensor_layout = key_tensor_layout;
      v_b = k_b;
      v_m = k_m;
      v_h = k_h;
      v_k = k_k;
    }

    if (attn_bias) {
      const auto attn_bias_shape = JUST(attn_bias)->shape();
      const int64_t num_attn_bias_axes = attn_bias_shape->NumAxes();
      CHECK_OR_RETURN(num_attn_bias_axes > 0 && num_attn_bias_axes <= 4)
          << "The number of dimensions of attn_bias should be greater than 0 and less than or "
             "equal to 4.";
      CHECK_GE_OR_RETURN(attn_bias_shape->At(num_attn_bias_axes - 1), k_m)
          << "The size of the -1 dimension of attn_bias should be greater than or equal to the "
             "dimension 'M' of the key tensor";
      CHECK_EQ_OR_RETURN(attn_bias_shape->At(num_attn_bias_axes - 1) % 8, 0)
          << "The size of the -1 dimension of attn_bias should be a multiple of 8.";
      if (num_attn_bias_axes >= 2) {
        CHECK_OR_RETURN(attn_bias_shape->At(num_attn_bias_axes - 2) == 1
                        || attn_bias_shape->At(num_attn_bias_axes - 2) >= q_m)
            << "The size of the -2 dimension of attn_bias should be greater than or equal to the "
               "dimension 'M' of the query tensor or equal to 1.";
      }
      if (num_attn_bias_axes >= 3) {
        CHECK_OR_RETURN(attn_bias_shape->At(num_attn_bias_axes - 3) == 1
                        || attn_bias_shape->At(num_attn_bias_axes - 3) == q_h)
            << "The size of the -3 dimension of attn_bias should be equal to the dimension 'H' of "
               "the query tensor or equal to 1.";
      }
      if (num_attn_bias_axes == 4) {
        CHECK_OR_RETURN(attn_bias_shape->At(0) == 1 || attn_bias_shape->At(0) == q_b)
            << "The size of the -4 dimension of attn_bias should be equal to the dimension 'B' of "
               "the query tensor or equal to 1.";
      }
    }

    std::string op_output_layout;
    if (output_layout == "BM(HK)") {
      op_output_layout = output_layout;
    } else if (output_layout == "MB(HK)") {
      if (q_b == 1) {
        op_output_layout = output_layout;
      } else {
        op_output_layout = "BM(HK)";
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "output_layout should be 'BM(HK)' or 'MB(HK)'";
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("query_layout", "key_layout", "value_layout",
                                                 "output_layout", "query_head_size", "causal",
                                                 "causal_diagonal_offset");
    attrs.SetAllAttrs(query_layout, key_tensor_layout, value_tensor_layout, op_output_layout, q_k,
                      causal, causal_diagonal_offset);
    std::shared_ptr<one::Tensor> op_output;
    if (attn_bias) {
      op_output = JUST(OpInterpUtil::Dispatch<Tensor>(
          *op_with_attn_bias_, {query, key_tensor, value_tensor, JUST(attn_bias)}, attrs));
    } else {
      op_output =
          JUST(OpInterpUtil::Dispatch<Tensor>(*op_, {query, key_tensor, value_tensor}, attrs));
    }
    if (op_output_layout == output_layout) {
      return op_output;
    } else {
      if (op_output_layout == "BM(HK)" && output_layout == "MB(HK)") {
        return functional::Transpose(op_output, {1, 0, 2});
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_with_attn_bias_;
};

class FusedAttentionConcatPastKeyValueFunctor {
 public:
  FusedAttentionConcatPastKeyValueFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("fused_attention_concat_past_key_value")
                         .Input("past_key")
                         .Input("past_value")
                         .Input("key")
                         .Input("value")
                         .Output("output_key")
                         .Output("output_value")
                         .Build());
  }
  Maybe<TensorTuple> operator()(
      const std::shared_ptr<one::Tensor>& past_key, const std::string& past_key_layout,
      const std::shared_ptr<one::Tensor>& past_value, const std::string& past_value_layout,
      const std::shared_ptr<one::Tensor>& key, const std::string& key_layout,
      const std::shared_ptr<one::Tensor>& value, const std::string& value_layout,
      const Optional<int64_t>& key_head_size) const {
    int64_t past_k_b = 0;
    int64_t past_k_m = 0;
    int64_t past_k_h = 0;
    int64_t past_k_k = 0;
    JUST(ParseDims("past_key", *past_key->shape(), past_key_layout, Optional<int64_t>(),
                   key_head_size, &past_k_b, &past_k_m, &past_k_h, &past_k_k));

    int64_t past_v_b = 0;
    int64_t past_v_m = 0;
    int64_t past_v_h = 0;
    int64_t past_v_k = 0;
    JUST(ParseDims("past_value", *past_value->shape(), past_value_layout, past_k_h, past_k_k,
                   &past_v_b, &past_v_m, &past_v_h, &past_v_k));
    CHECK_EQ_OR_RETURN(past_v_b, past_k_b) << "The size of dimension 'B' of the past_value tensor "
                                              "should be the same as that of the past_key tensor.";
    CHECK_EQ_OR_RETURN(past_v_m, past_k_m) << "The size of dimension 'M' of the past_value tensor "
                                              "should be the same as that of the past_key tensor.";

    int64_t k_b = 0;
    int64_t k_m = 0;
    int64_t k_h = 0;
    int64_t k_k = 0;
    JUST(ParseDims("key", *key->shape(), key_layout, past_k_h, past_k_k, &k_b, &k_m, &k_h, &k_k));
    CHECK_EQ_OR_RETURN(k_b, past_k_b) << "The size of dimension 'B' of the key tensor should be "
                                         "the same as that of the past_key tensor.";

    int64_t v_b = 0;
    int64_t v_m = 0;
    int64_t v_h = 0;
    int64_t v_k = 0;
    JUST(ParseDims("value", *value->shape(), value_layout, past_k_h, past_k_k, &v_b, &v_m, &v_h,
                   &v_k));
    CHECK_EQ_OR_RETURN(v_b, past_k_b) << "The size of dimension 'B' of the value tensor should be "
                                         "the same as that of the past_key tensor.";
    CHECK_EQ_OR_RETURN(v_m, k_m) << "The size of dimension 'M' of the value tensor should be the "
                                    "same as that of the key tensor.";

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("past_key_layout", "past_value_layout",
                                                 "key_layout", "value_layout", "key_head_size");
    attrs.SetAllAttrs(past_key_layout, past_value_layout, key_layout, value_layout, past_k_k);
    return JUST(
        OpInterpUtil::Dispatch<TensorTuple>(*op_, {past_key, past_value, key, value}, attrs));
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

}  // namespace impl

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor<impl::FusedMultiHeadAttentionInferenceFunctor>("FusedMultiHeadAttentionInference");
  m.add_functor<impl::FusedMultiHeadAttentionInferenceV2Functor>(
      "FusedMultiHeadAttentionInferenceV2");
  m.add_functor<impl::FusedAttentionConcatPastKeyValueFunctor>("FusedAttentionConcatPastKeyValue");
}

}  // namespace functional
}  // namespace one
}  // namespace oneflow
