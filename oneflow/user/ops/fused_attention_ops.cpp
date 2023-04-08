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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> ParseDims(const Shape& shape, const std::string& layout,
                      const Optional<int64_t>& batch_size, const Optional<int64_t>& seq_len,
                      const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
                      int64_t* b, int64_t* m, int64_t* h, int64_t* k, bool* bm_packed) {
  if (shape.NumAxes() == 2) {
    if (layout == "(BM)(HK)" || layout == "(BM)(H2K)" || layout == "(BM)(H3K)") {
      *bm_packed = true;
      CHECK_OR_RETURN(batch_size);
      CHECK_OR_RETURN(seq_len);
      *b = JUST(batch_size);
      *m = JUST(seq_len);
      int64_t packed_n = 0;
      if (layout == "(BM)(HK)") {
        packed_n = 1;
      } else if (layout == "(BM)(H2K)") {
        packed_n = 2;
      } else if (layout == "(BM)(H3K)") {
        packed_n = 3;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
      const int64_t hidden_size = shape.At(1);
      if (num_heads) {
        const int64_t expected_h = JUST(num_heads);
        const int64_t packed_h = packed_n * expected_h;
        CHECK_EQ_OR_RETURN(hidden_size % packed_h, 0);
        *h = expected_h;
        *k = hidden_size / packed_h;
      } else if (head_size) {
        const int64_t expected_k = JUST(head_size);
        const int64_t packed_k = packed_n * expected_k;
        CHECK_EQ_OR_RETURN(hidden_size % packed_k, 0);
        *h = hidden_size / packed_k;
        *k = expected_k;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  } else if (shape.NumAxes() == 3) {
    if (layout == "BM(HK)" || layout == "MB(HK)" || layout == "BM(H2K)" || layout == "MB(H2K)"
        || layout == "BM(H3K)" || layout == "MB(H3K)") {
      *bm_packed = false;
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
        *b = shape.At(0);
        *m = shape.At(1);
        packed_n = 2;
      } else if (layout == "MB(H2K)") {
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
        CHECK_EQ_OR_RETURN(hidden_size % packed_h, 0);
        *h = expected_h;
        *k = hidden_size / packed_h;
      } else if (head_size) {
        const int64_t expected_k = JUST(head_size);
        const int64_t packed_k = packed_n * expected_k;
        CHECK_EQ_OR_RETURN(hidden_size % packed_k, 0);
        *h = hidden_size / packed_k;
        *k = expected_k;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    } else if (layout == "(BM)HK") {
      *bm_packed = true;
      CHECK_OR_RETURN(batch_size);
      CHECK_OR_RETURN(seq_len);
      *b = JUST(batch_size);
      *m = JUST(seq_len);
      *h = shape.At(1);
      *k = shape.At(2);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  } else if (shape.NumAxes() == 4) {
    *bm_packed = false;
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
      UNIMPLEMENTED_THEN_RETURN();
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  };
  if (batch_size) {
    const int64_t expected_b = JUST(batch_size);
    CHECK_EQ_OR_RETURN(*b, expected_b);
  }
  if (seq_len) {
    const int64_t expected_m = JUST(seq_len);
    CHECK_EQ_OR_RETURN(*m, expected_m);
  }
  if (num_heads) {
    const int64_t expected_h = JUST(num_heads);
    CHECK_EQ_OR_RETURN(*h, expected_h);
  }
  if (head_size) {
    const int64_t expected_k = JUST(head_size);
    CHECK_EQ_OR_RETURN(*k, expected_k);
  }

  return Maybe<void>::Ok();
}

Maybe<void> ParseDims(const Shape& shape, const std::string& layout,
                      const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
                      int64_t* b, int64_t* m, int64_t* h, int64_t* k) {
  bool bm_packed{};
  return ParseDims(shape, layout, Optional<int64_t>(), Optional<int64_t>(), num_heads, head_size, b,
                   m, h, k, &bm_packed);
}

Maybe<Shape> LayoutToShape(int64_t b, int64_t m, int64_t h, int64_t k, const std::string& layout) {
  if (layout == "BM(HK)") {
    return Shape({b, m, h * k});
  } else if (layout == "BM(H2K)") {
    return Shape({b, m, h * k * 2});
  } else if (layout == "BM(H3K)") {
    return Shape({b, m, h * k * 3});
  } else if (layout == "MB(HK)") {
    return Shape({m, b, h * k});
  } else if (layout == "MB(H2K)") {
    return Shape({m, b, h * k * 2});
  } else if (layout == "MB(H3K)") {
    return Shape({m, b, h * k * 3});
  } else if (layout == "BMHK") {
    return Shape({b, m, h, k});
  } else if (layout == "BHMK") {
    return Shape({b, h, m, k});
  } else if (layout == "MBHK") {
    return Shape({m, b, h, k});
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
}

Maybe<void> ParseSplitAxis(const std::string& layout, bool can_hk_split, int64_t* b_split_axis,
                           int64_t* h_split_axis) {
  if (layout == "BM(HK)" || layout == "BM(H2K)" || layout == "BM(H3K)") {
    *b_split_axis = 0;
    if (can_hk_split) {
      *h_split_axis = 2;
    } else {
      *h_split_axis = -1;
    }
  } else if (layout == "MB(HK)" || layout == "MB(H2K)" || layout == "MB(H3K)") {
    *b_split_axis = 1;
    if (can_hk_split) {
      *h_split_axis = 2;
    } else {
      *h_split_axis = -1;
    }
  } else if (layout == "BMHK") {
    *b_split_axis = 0;
    *h_split_axis = 2;
  } else if (layout == "BHMK") {
    *b_split_axis = 0;
    *h_split_axis = 1;
  } else if (layout == "MBHK") {
    *b_split_axis = 1;
    *h_split_axis = 2;
  } else if (layout == "(BM)HK") {
    *b_split_axis = -1;
    *h_split_axis = 1;
  } else if (layout == "(BM)(HK)" || layout == "(BM)(H2K)" || layout == "(BM)(H3K)") {
    *b_split_axis = -1;
    if (can_hk_split) {
      *h_split_axis = 1;
    } else {
      *h_split_axis = -1;
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
};

}  // namespace

/*static*/ auto FusedMultiHeadAttentionInferenceOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType query_type = ctx->InputDType("query", 0);
  DataType key_type = ctx->InputDType("key", 0);
  DataType value_type = ctx->InputDType("value", 0);
  CHECK_EQ_OR_RETURN(key_type, query_type);
  CHECK_EQ_OR_RETURN(value_type, query_type);
  if (ctx->has_input("attn_bias", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("attn_bias", 0), query_type);
  }
  if (ctx->has_input("query_seq_start", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("query_seq_start", 0), DataType::kInt32);
  }
  if (ctx->has_input("key_seq_start", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("key_seq_start", 0), DataType::kInt32);
  }
  if (ctx->has_input("key_seq_len", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("key_seq_len", 0), DataType::kInt32);
  }
  ctx->SetOutputDType("out", 0, query_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMultiHeadAttentionInferenceOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  const int64_t query_head_size = ctx->Attr<int64_t>("query_head_size");
  CHECK_GE_OR_RETURN(query_head_size, 1);

  Optional<int64_t> batch_size;
  if (ctx->has_input("query_seq_start", 0)) {
    CHECK_OR_RETURN(ctx->has_input("key_seq_start", 0));
    const Shape& query_seq_start_shape = ctx->InputShape("query_seq_start", 0);
    CHECK_EQ_OR_RETURN(query_seq_start_shape.NumAxes(), 1);
    CHECK_GT_OR_RETURN(query_seq_start_shape.At(0), 1);
    CHECK_OR_RETURN(ctx->InputShape("key_seq_start", 0) == query_seq_start_shape);
    batch_size = query_seq_start_shape.At(0) - 1;
    if (ctx->has_input("key_seq_len", 0)) {
      const Shape& key_seq_len_shape = ctx->InputShape("key_seq_len", 0);
      CHECK_EQ_OR_RETURN(key_seq_len_shape.NumAxes(), 1);
      CHECK_EQ_OR_RETURN(key_seq_len_shape.At(0), query_seq_start_shape.At(0) - 1);
    }
  } else {
    CHECK_OR_RETURN(!ctx->has_input("key_seq_start", 0));
    CHECK_OR_RETURN(!ctx->has_input("key_seq_len", 0));
  }

  Optional<int64_t> query_max_seq_len;
  const int64_t attr_query_max_seq_len = ctx->Attr<int64_t>("query_max_seq_len");
  if (attr_query_max_seq_len != 0) { query_max_seq_len = attr_query_max_seq_len; }
  Optional<int64_t> key_max_seq_len;
  const int64_t attr_key_max_seq_len = ctx->Attr<int64_t>("key_max_seq_len");
  if (attr_key_max_seq_len != 0) { key_max_seq_len = attr_key_max_seq_len; }
  const Shape& query_shape = ctx->InputShape("query", 0);
  const std::string& query_layout = ctx->Attr<std::string>("query_layout");
  int64_t q_b = 0;
  int64_t q_m = 0;
  int64_t q_h = 0;
  int64_t q_k = 0;
  bool q_bm_packed = false;
  JUST(ParseDims(query_shape, query_layout, batch_size, query_max_seq_len, Optional<int64_t>(),
                 query_head_size, &q_b, &q_m, &q_h, &q_k, &q_bm_packed));
  if (q_bm_packed) { CHECK_OR_RETURN(ctx->has_input("query_seq_start", 0)); }

  const Shape& key_shape = ctx->InputShape("key", 0);
  const std::string& key_layout = ctx->Attr<std::string>("key_layout");
  int64_t k_b = 0;
  int64_t k_m = 0;
  int64_t k_h = 0;
  int64_t k_k = 0;
  bool k_bm_packed = false;
  JUST(ParseDims(key_shape, key_layout, q_b, key_max_seq_len, q_h, q_k, &k_b, &k_m, &k_h, &k_k,
                 &k_bm_packed));
  CHECK_EQ_OR_RETURN(k_b, q_b);
  CHECK_EQ_OR_RETURN(k_h, q_h);
  CHECK_EQ_OR_RETURN(k_bm_packed, q_bm_packed);

  const Shape& value_shape = ctx->InputShape("value", 0);
  const std::string& value_layout = ctx->Attr<std::string>("value_layout");
  int64_t v_b = 0;
  int64_t v_m = 0;
  int64_t v_h = 0;
  int64_t v_k = 0;
  bool v_bm_packed = false;
  JUST(ParseDims(value_shape, value_layout, q_b, k_m, q_h, Optional<int64_t>(), &v_b, &v_m, &v_h,
                 &v_k, &v_bm_packed));
  CHECK_EQ_OR_RETURN(v_b, q_b);
  CHECK_EQ_OR_RETURN(v_m, k_m);
  CHECK_EQ_OR_RETURN(v_bm_packed, k_bm_packed);

  if (ctx->has_input("attn_bias", 0)) {
    const Shape& attn_bias_shape = ctx->InputShape("attn_bias", 0);
    const int64_t num_attn_bias_axes = attn_bias_shape.NumAxes();
    CHECK_GE_OR_RETURN(num_attn_bias_axes, 1);
    CHECK_LE_OR_RETURN(num_attn_bias_axes, 4);
    DimVector padded_attn_bias_shape;
    for (int i = 0; i < 4 - num_attn_bias_axes; ++i) { padded_attn_bias_shape.push_back(1); }
    for (int i = 0; i < num_attn_bias_axes; ++i) {
      padded_attn_bias_shape.push_back(attn_bias_shape.At(i));
    }
    CHECK_OR_RETURN(padded_attn_bias_shape.at(0) == 1 || padded_attn_bias_shape.at(0) == q_b);
    CHECK_OR_RETURN(padded_attn_bias_shape.at(1) == 1 || padded_attn_bias_shape.at(1) == q_h);
    CHECK_OR_RETURN(padded_attn_bias_shape.at(2) == 1 || padded_attn_bias_shape.at(2) >= q_m);
    CHECK_OR_RETURN(padded_attn_bias_shape.at(3) >= k_m);
  }
  const std::string& output_layout = ctx->Attr<std::string>("output_layout");
  const bool o_bm_packed = output_layout == "(BM)(HK)";
  CHECK_EQ(o_bm_packed, q_bm_packed);
  if (output_layout == "(BM)(HK)") {
    ctx->SetOutputShape("out", 0, Shape({query_shape.At(0), q_h * v_k}));
  } else if (output_layout == "BM(HK)") {
    ctx->SetOutputShape("out", 0, Shape({q_b, q_m, q_h * v_k}));
  } else if (output_layout == "MB(HK)") {
    ctx->SetOutputShape("out", 0, Shape({q_m, q_b, q_h * v_k}));
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}
/*static*/ auto FusedMultiHeadAttentionInferenceOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return FusedMultiHeadAttentionInferenceOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedMultiHeadAttentionInferenceOp::GetSbp(user_op::SbpContext* ctx)
    -> Maybe<void> {
  const int64_t query_head_size = ctx->user_op_conf().attr<int64_t>("query_head_size");
  const std::string& query_layout = ctx->user_op_conf().attr<std::string>("query_layout");
  const std::string& key_layout = ctx->user_op_conf().attr<std::string>("key_layout");
  const std::string& value_layout = ctx->user_op_conf().attr<std::string>("value_layout");
  const std::string& output_layout = ctx->user_op_conf().attr<std::string>("output_layout");
  int64_t num_heads = 0;
  const user_op::TensorDesc& query = ctx->LogicalTensorDesc4InputArgNameAndIndex("query", 0);
  if (query.shape().NumAxes() == 2) {
    if (query_layout == "(BM)(HK)") {
      CHECK_EQ_OR_RETURN(query.shape().At(1) % query_head_size, 0);
      num_heads = query.shape().At(1) / query_head_size;
    } else if (query_layout == "(BM)(H3K)") {
      CHECK_EQ_OR_RETURN(query.shape().At(1) % (query_head_size * 3), 0);
      num_heads = query.shape().At(1) / (query_head_size * 3);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  } else if (query.shape().NumAxes() == 3) {
    if (query_layout == "BM(HK)" || query_layout == "MB(HK)") {
      CHECK_EQ_OR_RETURN(query.shape().At(2) % query_head_size, 0);
      num_heads = query.shape().At(2) / query_head_size;
    } else if (query_layout == "BM(H3K)" || query_layout == "MB(H3K)") {
      CHECK_EQ_OR_RETURN(query.shape().At(2) % (query_head_size * 3), 0);
      num_heads = query.shape().At(2) / (query_head_size * 3);
    } else if (query_layout == "(BM)HK") {
      num_heads = query.shape().At(1);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  } else if (query.shape().NumAxes() == 4) {
    if (query_layout == "BMHK") {
      num_heads = query.shape().At(2);
    } else if (query_layout == "BHMK") {
      num_heads = query.shape().At(1);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  const bool can_hk_split = num_heads % ctx->parallel_num() == 0;
  int64_t q_b_split_axis = -1;
  int64_t q_h_split_axis = -1;
  JUST(ParseSplitAxis(query_layout, can_hk_split, &q_b_split_axis, &q_h_split_axis));
  int64_t k_b_split_axis = -1;
  int64_t k_h_split_axis = -1;
  JUST(ParseSplitAxis(key_layout, can_hk_split, &k_b_split_axis, &k_h_split_axis));
  int64_t v_b_split_axis = -1;
  int64_t v_h_split_axis = -1;
  JUST(ParseSplitAxis(value_layout, can_hk_split, &v_b_split_axis, &v_h_split_axis));
  int64_t o_b_split_axis = -1;
  int64_t o_h_split_axis = -1;
  JUST(ParseSplitAxis(output_layout, can_hk_split, &o_b_split_axis, &o_h_split_axis));

  std::vector<user_op::OpArg> attn_bias_arg;
  if (ctx->user_op_conf().has_input("attn_bias", 0)) { attn_bias_arg.emplace_back("attn_bias", 0); }
  std::vector<user_op::OpArg> var_len_args;
  if (ctx->user_op_conf().has_input("query_seq_start", 0)) {
    var_len_args.emplace_back("query_seq_start", 0);
  }
  if (ctx->user_op_conf().has_input("key_seq_start", 0)) {
    var_len_args.emplace_back("key_seq_start", 0);
  }
  if (ctx->user_op_conf().has_input("key_seq_len", 0)) {
    var_len_args.emplace_back("key_seq_len", 0);
  }
  if (q_b_split_axis >= 0 && k_b_split_axis >= 0 && v_b_split_axis >= 0 && o_b_split_axis >= 0
      && var_len_args.empty()) {
    bool broadcast_attn_bias = false;
    if (ctx->user_op_conf().has_input("attn_bias", 0)) {
      const user_op::TensorDesc& attn_bias =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("attn_bias", 0);
      if (attn_bias.shape().NumAxes() < 4 || attn_bias.shape().At(0) == 1) {
        broadcast_attn_bias = true;
      }
    }
    if (broadcast_attn_bias) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("query", 0), q_b_split_axis)
          .Split(user_op::OpArg("key", 0), k_b_split_axis)
          .Split(user_op::OpArg("value", 0), v_b_split_axis)
          .Broadcast(attn_bias_arg)
          .Split(ctx->outputs(), o_b_split_axis)
          .Build();

    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("query", 0), q_b_split_axis)
          .Split(user_op::OpArg("key", 0), k_b_split_axis)
          .Split(user_op::OpArg("value", 0), v_b_split_axis)
          .Split(attn_bias_arg, 0)
          .Split(ctx->outputs(), o_b_split_axis)
          .Build();
    }
  }
  if (q_h_split_axis >= 0 && k_h_split_axis >= 0 && v_h_split_axis >= 0 && o_h_split_axis >= 0) {
    bool broadcast_attn_bias = false;
    if (ctx->user_op_conf().has_input("attn_bias", 0)) {
      const user_op::TensorDesc& attn_bias =
          ctx->LogicalTensorDesc4InputArgNameAndIndex("attn_bias", 0);
      if (attn_bias.shape().NumAxes() == 4) {
        if (attn_bias.shape().At(1) == 1) { broadcast_attn_bias = true; }
      } else if (attn_bias.shape().NumAxes() == 3) {
        if (attn_bias.shape().At(0) == 1) { broadcast_attn_bias = true; }
      } else {
        broadcast_attn_bias = true;
      }
    }
    if (broadcast_attn_bias) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("query", 0), q_h_split_axis)
          .Split(user_op::OpArg("key", 0), k_h_split_axis)
          .Split(user_op::OpArg("value", 0), v_h_split_axis)
          .Broadcast(attn_bias_arg)
          .Broadcast(var_len_args)
          .Split(ctx->outputs(), o_h_split_axis)
          .Build();

    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("query", 0), q_h_split_axis)
          .Split(user_op::OpArg("key", 0), k_h_split_axis)
          .Split(user_op::OpArg("value", 0), v_h_split_axis)
          .Split(attn_bias_arg, 1)
          .Broadcast(var_len_args)
          .Split(ctx->outputs(), o_h_split_axis)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedAttentionConcatPastKeyValueOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  const DataType data_type = ctx->InputDType("key", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("value", 0), data_type);
  if (ctx->has_input("past_key", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("past_key", 0), data_type);
  }
  if (ctx->has_input("past_value", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("past_value", 0), data_type);
  }
  ctx->SetOutputDType("output_key", 0, data_type);
  ctx->SetOutputDType("output_value", 0, data_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedAttentionConcatPastKeyValueOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  const int64_t key_head_size = ctx->Attr<int64_t>("key_head_size");
  CHECK_GE_OR_RETURN(key_head_size, 1);

  const Shape& key_shape = ctx->InputShape("key", 0);
  const std::string& key_layout = ctx->Attr<std::string>("key_layout");
  int64_t k_b = 0;
  int64_t k_m = 0;
  int64_t k_h = 0;
  int64_t k_k = 0;
  JUST(
      ParseDims(key_shape, key_layout, Optional<int64_t>(), key_head_size, &k_b, &k_m, &k_h, &k_k));

  const Shape& value_shape = ctx->InputShape("value", 0);
  const std::string& value_layout = ctx->Attr<std::string>("value_layout");
  int64_t v_b = 0;
  int64_t v_m = 0;
  int64_t v_h = 0;
  int64_t v_k = 0;
  JUST(ParseDims(value_shape, value_layout, k_h, k_k, &v_b, &v_m, &v_h, &v_k));
  CHECK_EQ_OR_RETURN(v_b, k_b);
  CHECK_EQ_OR_RETURN(v_m, k_m);

  int64_t past_k_b = 0;
  int64_t past_k_m = 0;
  int64_t past_k_h = 0;
  int64_t past_k_k = 0;
  int64_t past_v_b = 0;
  int64_t past_v_m = 0;
  int64_t past_v_h = 0;
  int64_t past_v_k = 0;
  const std::string& past_key_layout = ctx->Attr<std::string>("past_key_layout");
  const std::string& past_value_layout = ctx->Attr<std::string>("past_value_layout");
  if (ctx->has_input("past_key", 0)) {
    CHECK_OR_RETURN(ctx->has_input("past_value", 0));
    const Shape& past_key_shape = ctx->InputShape("past_key", 0);
    JUST(ParseDims(past_key_shape, past_key_layout, k_h, k_k, &past_k_b, &past_k_m, &past_k_h,
                   &past_k_k));
    CHECK_EQ_OR_RETURN(past_k_b, k_b);

    const Shape& past_value_shape = ctx->InputShape("past_value", 0);
    JUST(ParseDims(past_value_shape, past_value_layout, k_h, k_k, &past_v_b, &past_v_m, &past_v_h,
                   &past_v_k));
    CHECK_EQ_OR_RETURN(past_v_b, k_b);
    CHECK_EQ_OR_RETURN(past_v_m, past_k_m);
  } else {
    CHECK_OR_RETURN(!ctx->has_input("past_value", 0));
  }

  ctx->SetOutputShape("output_key", 0,
                      *JUST(LayoutToShape(k_b, past_k_m + k_m, k_h, k_k, past_key_layout)));
  ctx->SetOutputShape("output_value", 0,
                      *JUST(LayoutToShape(v_b, past_v_m + v_m, v_h, v_k, past_value_layout)));
  return Maybe<void>::Ok();
}
/*static*/ auto FusedAttentionConcatPastKeyValueOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return FusedAttentionConcatPastKeyValueOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FusedAttentionConcatPastKeyValueOp::GetSbp(user_op::SbpContext* ctx)
    -> Maybe<void> {
  const int64_t key_head_size = ctx->user_op_conf().attr<int64_t>("key_head_size");
  const std::string& past_key_layout = ctx->user_op_conf().attr<std::string>("past_key_layout");
  const std::string& past_value_layout = ctx->user_op_conf().attr<std::string>("past_value_layout");
  const std::string& key_layout = ctx->user_op_conf().attr<std::string>("key_layout");
  const std::string& value_layout = ctx->user_op_conf().attr<std::string>("value_layout");
  int64_t num_heads = 0;
  {
    int64_t b = 0;
    int64_t m = 0;
    int64_t k = 0;

    const user_op::TensorDesc& key = ctx->LogicalTensorDesc4InputArgNameAndIndex("key", 0);
    JUST(ParseDims(key.shape(), key_layout, Optional<int64_t>(), key_head_size, &b, &m, &num_heads,
                   &k));
  }
  const bool can_hk_split = num_heads % ctx->parallel_num() == 0;
  int64_t past_k_b_split_axis = -1;
  int64_t past_k_h_split_axis = -1;
  JUST(ParseSplitAxis(past_key_layout, can_hk_split, &past_k_b_split_axis, &past_k_h_split_axis));
  int64_t past_v_b_split_axis = -1;
  int64_t past_v_h_split_axis = -1;
  JUST(ParseSplitAxis(past_value_layout, can_hk_split, &past_v_b_split_axis, &past_v_h_split_axis));
  int64_t k_b_split_axis = -1;
  int64_t k_h_split_axis = -1;
  JUST(ParseSplitAxis(key_layout, can_hk_split, &k_b_split_axis, &k_h_split_axis));
  int64_t v_b_split_axis = -1;
  int64_t v_h_split_axis = -1;
  JUST(ParseSplitAxis(value_layout, can_hk_split, &v_b_split_axis, &v_h_split_axis));

  std::vector<user_op::OpArg> past_key_arg;
  if (ctx->user_op_conf().has_input("past_key", 0)) { past_key_arg.emplace_back("past_key", 0); }
  std::vector<user_op::OpArg> past_value_arg;
  if (ctx->user_op_conf().has_input("past_value", 0)) {
    past_value_arg.emplace_back("past_value", 0);
  }
  if (past_k_b_split_axis >= 0 && past_v_b_split_axis >= 0 && k_b_split_axis >= 0
      && v_b_split_axis >= 0) {
    ctx->NewBuilder()
        .Split(past_key_arg, past_k_b_split_axis)
        .Split(past_value_arg, past_v_b_split_axis)
        .Split(user_op::OpArg("key", 0), k_b_split_axis)
        .Split(user_op::OpArg("value", 0), v_b_split_axis)
        .Split(user_op::OpArg("output_key", 0), past_k_b_split_axis)
        .Split(user_op::OpArg("output_value", 0), past_v_b_split_axis)
        .Build();
  }

  if (past_k_h_split_axis >= 0 && past_v_h_split_axis >= 0 && k_h_split_axis >= 0
      && v_h_split_axis >= 0) {
    ctx->NewBuilder()
        .Split(past_key_arg, past_k_h_split_axis)
        .Split(past_value_arg, past_v_h_split_axis)
        .Split(user_op::OpArg("key", 0), k_h_split_axis)
        .Split(user_op::OpArg("value", 0), v_h_split_axis)
        .Split(user_op::OpArg("output_key", 0), past_k_h_split_axis)
        .Split(user_op::OpArg("output_value", 0), past_v_h_split_axis)
        .Build();
  }

  return Maybe<void>::Ok();
}

}  // namespace oneflow
