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

#include "oneflow/core/common/scalar.h"
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/mutable_attr_map.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_interpreter/lazy_op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/functional/functional.h"
#include "oneflow/core/functional/function_library.h"

namespace oneflow {
namespace one {
namespace functional {

namespace impl {

class OneEmbeddingIdShuffleFunctor {
 public:
  OneEmbeddingIdShuffleFunctor() {
    op_table_ids_has_in_out_ = CHECK_JUST(one::OpBuilder("id_shuffle")
                                              .Input("ids")
                                              .Input("table_ids")
                                              .Output("num_unique_matrix")
                                              .Output("inverse_unique_partition_indices")
                                              .Output("cur_rank_num_unique")
                                              .Output("cur_rank_unique_ids")
                                              .Output("cur_rank_unique_table_ids")
                                              .Output("cur_rank_inverse_indices")
                                              .Build());
    op_table_ids_no_in_has_out_ = CHECK_JUST(one::OpBuilder("id_shuffle")
                                                 .Input("ids")
                                                 .Output("num_unique_matrix")
                                                 .Output("inverse_unique_partition_indices")
                                                 .Output("cur_rank_num_unique")
                                                 .Output("cur_rank_unique_ids")
                                                 .Output("cur_rank_unique_table_ids")
                                                 .Output("cur_rank_inverse_indices")
                                                 .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& ids,
                                const Optional<one::Tensor>& table_ids, const int32_t& num_tables,
                                const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("num_tables", "embedding_name");
    attrs.SetAllAttrs(num_tables, embedding_name);
    if (table_ids) {
      return OpInterpUtil::Dispatch<TensorTuple>(*op_table_ids_has_in_out_, {ids, JUST(table_ids)},
                                                 attrs);
    } else {
      return OpInterpUtil::Dispatch<TensorTuple>(*op_table_ids_no_in_has_out_, {ids}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_table_ids_has_in_out_;
  std::shared_ptr<OpExpr> op_table_ids_no_in_has_out_;
};

class OneEmbeddingEmbeddingShuffleFunctor {
 public:
  OneEmbeddingEmbeddingShuffleFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_shuffle")
                         .Input("cur_rank_embeddings")
                         .Input("num_unique_matrix")
                         .Input("cur_rank_inverse_indices")
                         .Input("inverse_unique_partition_indices")
                         .Output("embeddings")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& cur_rank_embeddings,
                           const std::shared_ptr<one::Tensor>& num_unique_matrix,
                           const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices,
                           const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices,
                           const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("embedding_size", "embedding_name");
    const int64_t num_axes = cur_rank_embeddings->shape()->NumAxes();
    attrs.SetAllAttrs(cur_rank_embeddings->shape()->At(num_axes - 1), embedding_name);
    return OpInterpUtil::Dispatch<Tensor>(
        *op_,
        {cur_rank_embeddings, num_unique_matrix, cur_rank_inverse_indices,
         inverse_unique_partition_indices},
        attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingEmbeddingGradientShuffleFunctor {
 public:
  OneEmbeddingEmbeddingGradientShuffleFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_gradient_shuffle")
                         .Input("embedding_grad")
                         .Input("num_unique_matrix")
                         .Input("cur_rank_inverse_indices")
                         .Input("inverse_unique_partition_indices")
                         .Output("cur_rank_unique_embedding_grad")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& embedding_grad,
                           const std::shared_ptr<one::Tensor>& num_unique_matrix,
                           const std::shared_ptr<one::Tensor>& cur_rank_inverse_indices,
                           const std::shared_ptr<one::Tensor>& inverse_unique_partition_indices,
                           const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("embedding_size", "embedding_name");
    const int64_t num_axes = embedding_grad->shape()->NumAxes();
    attrs.SetAllAttrs(embedding_grad->shape()->At(num_axes - 1), embedding_name);
    return OpInterpUtil::Dispatch<Tensor>(
        *op_,
        {embedding_grad, num_unique_matrix, cur_rank_inverse_indices,
         inverse_unique_partition_indices},
        attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingLookupFunctor {
 public:
  OneEmbeddingLookupFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_lookup")
                         .Input("num_unique_ids")
                         .Input("unique_ids")
                         .Input("table_ids")
                         .Output("unique_values")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_ids,
                           const std::shared_ptr<one::Tensor>& table_ids,
                           const Symbol<DType>& dtype, const Symbol<DType>& embedding_dtype,
                           const int64_t line_size, const int64_t embedding_size,
                           const std::string& embedding_name, const std::string& embedding_tables,
                           const std::string& state_initializer, const int64_t seed) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("dtype", "embedding_dtype", "line_size",
                                                 "embedding_size", "embedding_name",
                                                 "embedding_tables", "state_initializer", "seed");
    attrs.SetAllAttrs(dtype->data_type(), embedding_dtype->data_type(), line_size, embedding_size,
                      embedding_name, embedding_tables, state_initializer, seed);
    return OpInterpUtil::Dispatch<Tensor>(*op_, {num_unique_ids, unique_ids, table_ids}, attrs);
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingFusedLookupFunctor {
 public:
  OneEmbeddingFusedLookupFunctor() {
    op_has_table_ids_ = CHECK_JUST(one::OpBuilder("one_embedding_fused_lookup")
                                       .Input("shadow")
                                       .Input("ids")
                                       .Input("table_ids")
                                       .Output("embeddings")
                                       .Build());
    op_no_table_ids_ = CHECK_JUST(one::OpBuilder("one_embedding_fused_lookup")
                                      .Input("shadow")
                                      .Input("ids")
                                      .Output("embeddings")
                                      .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& shadow,
                           const std::shared_ptr<one::Tensor>& ids,
                           const Optional<one::Tensor>& table_ids, const Symbol<DType>& dtype,
                           const std::string& embedding_name, const int64_t line_size,
                           const int64_t embedding_size, const bool is_full_cache,
                           const int32_t num_tables, const std::string& embedding_tables,
                           const Optional<int64_t>& padding_idx, const int64_t seed) const {
    int64_t padding_idx_val = -1;
    bool has_padding_idx = false;
    if (padding_idx.has_value()) {
      padding_idx_val = JUST(padding_idx);
      has_padding_idx = true;
    }
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
        "dtype", "embedding_name", "line_size", "embedding_size", "is_full_cache", "num_tables",
        "embedding_tables", "seed", "padding_idx", "has_padding_idx");
    attrs.SetAllAttrs(dtype->data_type(), embedding_name, line_size, embedding_size, is_full_cache,
                      num_tables, embedding_tables, seed, padding_idx_val, has_padding_idx);
    if (table_ids) {
      const auto& table_ids_shape = *(JUST(table_ids)->shape());
      const auto& ids_shape = *(ids->shape());
      auto broadcast_table_ids = JUST(table_ids);
      if (table_ids_shape != ids_shape) {
        CHECK_LE_OR_RETURN(table_ids_shape.NumAxes(), ids_shape.NumAxes())
            << "table_ids num_axes should be less equal to ids num_axes, but got table_ids "
               "num_axes "
            << table_ids_shape.NumAxes() << " and ids num_axes " << ids_shape.NumAxes();
        const int64_t left_extend_dims = ids_shape.NumAxes() - table_ids_shape.NumAxes();
        for (int64_t i = 0; i < table_ids_shape.NumAxes(); i++) {
          CHECK_EQ_OR_RETURN(table_ids_shape.at(i), ids_shape.at(left_extend_dims + i))
              << "when table_ids's shape not equals ids shape, table_ids must be able to be "
                 "broadcast to ids_shape "
                 "but got table_ids_shape: "
              << table_ids_shape.DebugStr() << ", ids_shape: " << ids_shape.DebugStr();
        }
        broadcast_table_ids =
            JUST(functional::BroadcastLike(JUST(table_ids), ids, std::vector<int32_t>{}));
      }
      return OpInterpUtil::Dispatch<Tensor>(*op_has_table_ids_, {shadow, ids, broadcast_table_ids},
                                            attrs);
    } else {
      return OpInterpUtil::Dispatch<Tensor>(*op_no_table_ids_, {shadow, ids}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_has_table_ids_;
  std::shared_ptr<OpExpr> op_no_table_ids_;
};

class OneEmbeddingEmbeddingPutFunctor {
 public:
  OneEmbeddingEmbeddingPutFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("embedding_put")
                         .Input("num_unique_ids")
                         .Input("unique_ids")
                         .Input("unique_embeddings")
                         .Build());
  }

  Maybe<void> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                         const std::shared_ptr<one::Tensor>& unique_ids,
                         const std::shared_ptr<one::Tensor>& unique_embeddings,
                         const std::string& embedding_name, const int64_t line_size) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("embedding_name", "line_size");
    attrs.SetAllAttrs(embedding_name, line_size);
    JUST(OpInterpUtil::Dispatch<TensorTuple>(*op_, {num_unique_ids, unique_ids, unique_embeddings},
                                             attrs));
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> op_;
};

class OneEmbeddingUniqueKeyValuePairFunctor {
 public:
  OneEmbeddingUniqueKeyValuePairFunctor() {
    op_has_input_value_ = CHECK_JUST(one::OpBuilder("unique_key_value_pair")
                                         .Input("keys")
                                         .Input("values")
                                         .Output("num_unique")
                                         .Output("unique_keys")
                                         .Output("unique_values")
                                         .Output("inverse_indices")
                                         .Build());
    op_no_input_value_ = CHECK_JUST(one::OpBuilder("unique_key_value_pair")
                                        .Input("keys")
                                        .Output("num_unique")
                                        .Output("unique_keys")
                                        .Output("unique_values")
                                        .Output("inverse_indices")
                                        .Build());
  }

  Maybe<TensorTuple> operator()(const std::shared_ptr<one::Tensor>& keys,
                                const Optional<one::Tensor>& values, const int32_t num_tables,
                                const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("num_tables", "embedding_name");
    attrs.SetAllAttrs(num_tables, embedding_name);
    if (values) {
      return OpInterpUtil::Dispatch<TensorTuple>(*op_has_input_value_, {keys, JUST(values)}, attrs);
    } else {
      return OpInterpUtil::Dispatch<TensorTuple>(*op_no_input_value_, {keys}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_has_input_value_;
  std::shared_ptr<OpExpr> op_no_input_value_;
};

class OneEmbeddingSgdUpdateFunctor {
 public:
  OneEmbeddingSgdUpdateFunctor() {
    // This functor is only used in one_embedding eager mode with lr passed by attr and no optional
    // input, we also define functor with all optional input just for unittest. when the optional
    // input learning_rate tensor has passed in, we think all optional input are not None and check
    // them.
    sgd_no_optional_input_op_ = CHECK_JUST(one::OpBuilder("one_embedding_sgd_update")
                                               .Input("num_unique_ids")
                                               .Input("unique_embeddings")
                                               .Input("embedding_grad")
                                               .Output("updated_unique_embeddings")
                                               .Build());
    momentum_no_optional_input_op_ = CHECK_JUST(one::OpBuilder("one_embedding_momentum_update")
                                                    .Input("num_unique_ids")
                                                    .Input("unique_embeddings")
                                                    .Input("embedding_grad")
                                                    .Output("updated_unique_embeddings")
                                                    .Build());
    // This functor is just for unittest
    sgd_op_ = CHECK_JUST(one::OpBuilder("one_embedding_sgd_update")
                             .Input("num_unique_ids")
                             .Input("unique_embeddings")
                             .Input("embedding_grad")
                             .Input("learning_rate")
                             .Input("down_scale_by_tensor")
                             .Input("skip_if")
                             .Output("updated_unique_embeddings")
                             .Build());
    momentum_op_ = CHECK_JUST(one::OpBuilder("one_embedding_momentum_update")
                                  .Input("num_unique_ids")
                                  .Input("unique_embeddings")
                                  .Input("embedding_grad")
                                  .Input("learning_rate")
                                  .Input("down_scale_by_tensor")
                                  .Input("skip_if")
                                  .Output("updated_unique_embeddings")
                                  .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const Optional<one::Tensor>& learning_rate,
                           const Optional<one::Tensor>& down_scale_by_tensor,
                           const Optional<one::Tensor>& skip_if, const float learning_rate_val,
                           const double scale, const float weight_decay, const float momentum,
                           const int64_t line_size, const int64_t embedding_size,
                           const std::string& embedding_name) const {
    auto& attrs =
        THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "weight_decay", "line_size",
                                       "embedding_size", "embedding_name", "beta");
    if (momentum == 0) {
      attrs.SetAllAttrs(learning_rate_val, scale, weight_decay, line_size, embedding_size,
                        embedding_name, NullOpt);

      if (learning_rate) {
        CHECK(down_scale_by_tensor);
        CHECK(skip_if);
        return OpInterpUtil::Dispatch<Tensor>(
            *sgd_op_,
            {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
             JUST(down_scale_by_tensor), JUST(skip_if)},
            attrs);
      } else {
        CHECK(!down_scale_by_tensor);
        CHECK(!skip_if);
        return OpInterpUtil::Dispatch<Tensor>(
            *sgd_no_optional_input_op_, {num_unique_ids, unique_embeddings, embedding_grad}, attrs);
      }
    } else {
      attrs.SetAllAttrs(learning_rate_val, scale, weight_decay, line_size, embedding_size,
                        embedding_name, momentum);
      if (learning_rate) {
        CHECK(down_scale_by_tensor);
        CHECK(skip_if);
        return OpInterpUtil::Dispatch<Tensor>(
            *momentum_op_,
            {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
             JUST(down_scale_by_tensor), JUST(skip_if)},
            attrs);
      } else {
        CHECK(!down_scale_by_tensor);
        CHECK(!skip_if);
        return OpInterpUtil::Dispatch<Tensor>(*momentum_no_optional_input_op_,
                                              {num_unique_ids, unique_embeddings, embedding_grad},
                                              attrs);
      }
    }
  }

 private:
  std::shared_ptr<OpExpr> sgd_no_optional_input_op_;
  std::shared_ptr<OpExpr> sgd_op_;
  std::shared_ptr<OpExpr> momentum_no_optional_input_op_;
  std::shared_ptr<OpExpr> momentum_op_;
};

class OneEmbeddingAdamUpdateFunctor {
 public:
  OneEmbeddingAdamUpdateFunctor() {
    // This functor is only used in one_embedding eager mode with lr passed by attr and no optional
    // input, we also define functor with all optional input just for unittest. when the optional
    // input learning_rate tensor has passed in, we think all optional input are not None and check
    // them.
    no_optional_input_op_ = CHECK_JUST(one::OpBuilder("one_embedding_adam_update")
                                           .Input("num_unique_ids")
                                           .Input("unique_embeddings")
                                           .Input("embedding_grad")
                                           .Output("updated_unique_embeddings")
                                           .Build());
    // This functor is just for unittest
    no_bias_correction_op_ = CHECK_JUST(one::OpBuilder("one_embedding_adam_update")
                                            .Input("num_unique_ids")
                                            .Input("unique_embeddings")
                                            .Input("embedding_grad")
                                            .Input("learning_rate")
                                            .Input("down_scale_by_tensor")
                                            .Input("skip_if")
                                            .Output("updated_unique_embeddings")
                                            .Build());
    do_bias_correction_op_ = CHECK_JUST(one::OpBuilder("one_embedding_adam_update")
                                            .Input("num_unique_ids")
                                            .Input("unique_embeddings")
                                            .Input("embedding_grad")
                                            .Input("learning_rate")
                                            .Input("down_scale_by_tensor")
                                            .Input("skip_if")
                                            .Input("bias_correction1")
                                            .Input("bias_correction2")
                                            .Output("updated_unique_embeddings")
                                            .Build());
  }

  Maybe<Tensor> operator()(
      const std::shared_ptr<one::Tensor>& num_unique_ids,
      const std::shared_ptr<one::Tensor>& unique_embeddings,
      const std::shared_ptr<one::Tensor>& embedding_grad,
      const Optional<one::Tensor>& learning_rate, const Optional<one::Tensor>& down_scale_by_tensor,
      const Optional<one::Tensor>& skip_if, const Optional<one::Tensor>& bias_correction1,
      const Optional<one::Tensor>& bias_correction2, const float learning_rate_val,
      const double scale, const float weight_decay, const float beta1, const float beta2,
      const float& bias_correction1_val, const float& bias_correction2_val, const float epsilon,
      const bool do_bias_correction, const int64_t line_size, const int64_t embedding_size,
      const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
        "learning_rate_val", "scale", "weight_decay", "beta1", "beta2", "epsilon",
        "bias_correction1_val", "bias_correction2_val", "do_bias_correction", "line_size",
        "embedding_size", "embedding_name");
    attrs.SetAllAttrs(learning_rate_val, scale, weight_decay, beta1, beta2, epsilon,
                      bias_correction1_val, bias_correction2_val, do_bias_correction, line_size,
                      embedding_size, embedding_name);
    if (learning_rate) {
      CHECK(down_scale_by_tensor);
      CHECK(skip_if);
      if (do_bias_correction) {
        CHECK(bias_correction1);
        CHECK(bias_correction2);
        return OpInterpUtil::Dispatch<Tensor>(
            *do_bias_correction_op_,
            {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
             JUST(down_scale_by_tensor), JUST(skip_if), JUST(bias_correction1),
             JUST(bias_correction2)},
            attrs);
      } else {
        return OpInterpUtil::Dispatch<Tensor>(
            *no_bias_correction_op_,
            {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
             JUST(down_scale_by_tensor), JUST(skip_if)},
            attrs);
      }
    } else {
      CHECK(!down_scale_by_tensor);
      CHECK(!skip_if);
      CHECK(!bias_correction1);
      CHECK(!bias_correction2);
      return OpInterpUtil::Dispatch<Tensor>(
          *no_optional_input_op_, {num_unique_ids, unique_embeddings, embedding_grad}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> no_bias_correction_op_;
  std::shared_ptr<OpExpr> do_bias_correction_op_;
  std::shared_ptr<OpExpr> no_optional_input_op_;
};

class OneEmbeddingAdagradUpdateFunctor {
 public:
  OneEmbeddingAdagradUpdateFunctor() {
    // This functor is only used in one_embedding eager mode with lr passed by attr and no optional
    // input, we also define functor with all optional input just for unittest. when the optional
    // input learning_rate tensor has passed in, we think all optional input are not None and check
    // them.
    op_no_optional_input_ = CHECK_JUST(one::OpBuilder("one_embedding_adagrad_update")
                                           .Input("num_unique_ids")
                                           .Input("unique_embeddings")
                                           .Input("embedding_grad")
                                           .Output("updated_unique_embeddings")
                                           .Build());
    // This functor is just for unittest
    op_ = CHECK_JUST(one::OpBuilder("one_embedding_adagrad_update")
                         .Input("num_unique_ids")
                         .Input("unique_embeddings")
                         .Input("embedding_grad")
                         .Input("learning_rate")
                         .Input("down_scale_by_tensor")
                         .Input("skip_if")
                         .Input("train_step")
                         .Output("updated_unique_embeddings")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const Optional<one::Tensor>& learning_rate,
                           const Optional<one::Tensor>& down_scale_by_tensor,
                           const Optional<one::Tensor>& skip_if,
                           const Optional<one::Tensor>& train_step, const int64_t train_step_val,
                           const float learning_rate_val, const double scale,
                           const float weight_decay, const float lr_decay, const float epsilon,
                           const int64_t line_size, const int64_t embedding_size,
                           const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("train_step_val", "learning_rate_val", "scale",
                                                 "weight_decay", "lr_decay", "epsilon", "line_size",
                                                 "embedding_size", "embedding_name");
    attrs.SetAllAttrs(train_step_val, learning_rate_val, scale, weight_decay, lr_decay, epsilon,
                      line_size, embedding_size, embedding_name);
    if (learning_rate) {
      CHECK(down_scale_by_tensor);
      CHECK(skip_if);
      CHECK(train_step);
      return OpInterpUtil::Dispatch<Tensor>(
          *op_,
          {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
           JUST(down_scale_by_tensor), JUST(skip_if), JUST(train_step)},
          attrs);
    } else {
      CHECK(!down_scale_by_tensor);
      CHECK(!skip_if);
      CHECK(!train_step);
      return OpInterpUtil::Dispatch<Tensor>(
          *op_no_optional_input_, {num_unique_ids, unique_embeddings, embedding_grad}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_no_optional_input_;
};

class OneEmbeddingFtrlUpdateFunctor {
 public:
  OneEmbeddingFtrlUpdateFunctor() {
    // This functor is only used in one_embedding eager mode with lr passed by attr and no optional
    // input, we also define functor with all optional input just for unittest. when the optional
    // input learning_rate tensor has passed in, we think all optional input are not None and check
    // them.
    op_no_optional_input_ = CHECK_JUST(one::OpBuilder("one_embedding_ftrl_update")
                                           .Input("num_unique_ids")
                                           .Input("unique_embeddings")
                                           .Input("embedding_grad")
                                           .Output("updated_unique_embeddings")
                                           .Build());
    // This functor is just for unittest
    op_ = CHECK_JUST(one::OpBuilder("one_embedding_ftrl_update")
                         .Input("num_unique_ids")
                         .Input("unique_embeddings")
                         .Input("embedding_grad")
                         .Input("learning_rate")
                         .Input("down_scale_by_tensor")
                         .Input("skip_if")
                         .Output("updated_unique_embeddings")
                         .Build());
  }

  Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& num_unique_ids,
                           const std::shared_ptr<one::Tensor>& unique_embeddings,
                           const std::shared_ptr<one::Tensor>& embedding_grad,
                           const Optional<one::Tensor>& learning_rate,
                           const Optional<one::Tensor>& down_scale_by_tensor,
                           const Optional<one::Tensor>& skip_if, const float learning_rate_val,
                           const double scale, const float weight_decay, const float lr_power,
                           const float lambda1, const float lambda2, const float beta,
                           const int64_t line_size, const int64_t embedding_size,
                           const std::string& embedding_name) const {
    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "weight_decay",
                                                 "lr_power", "lambda1", "lambda2", "beta",
                                                 "line_size", "embedding_size", "embedding_name");
    attrs.SetAllAttrs(learning_rate_val, scale, weight_decay, lr_power, lambda1, lambda2, beta,
                      line_size, embedding_size, embedding_name);
    if (learning_rate) {
      CHECK(down_scale_by_tensor);
      CHECK(skip_if);
      return OpInterpUtil::Dispatch<Tensor>(
          *op_,
          {num_unique_ids, unique_embeddings, embedding_grad, JUST(learning_rate),
           JUST(down_scale_by_tensor), JUST(skip_if)},
          attrs);
    } else {
      CHECK(!down_scale_by_tensor);
      CHECK(!skip_if);
      return OpInterpUtil::Dispatch<Tensor>(
          *op_no_optional_input_, {num_unique_ids, unique_embeddings, embedding_grad}, attrs);
    }
  }

 private:
  std::shared_ptr<OpExpr> op_;
  std::shared_ptr<OpExpr> op_no_optional_input_;
};

ONEFLOW_FUNCTION_LIBRARY(m) {
  m.add_functor(
      "DispatchFeedInput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        const auto& origin_input = JUST(OpInterpUtil::Dispatch<Tensor>(*op, {input}));
        // Unpack input when do grad acc
        return GradAccTryInsertUnpackAfterInput(origin_input);
      });
  m.add_functor(
      "DispatchFetchOutput",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input) -> Maybe<Tensor> {
        // Pack output when do grad acc
        const auto& pack_input = JUST(GradAccTryInsertPackBeforeOutput(input));
        return OpInterpUtil::Dispatch<Tensor>(*op, {pack_input});
      });
  m.add_functor("DispatchFeedVariable",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const Scalar& l2) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("l2");
                  attrs.SetAllAttrs(l2.As<double>());
                  const auto& origin_var =
                      JUST(OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs));
                  // Repeat variable when do grad acc
                  return GradAccTryInsertRepeatAfterVar(origin_var);
                });
  m.add_functor(
      "DispatchOfrecordReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_dir, int32_t data_part_num,
         const std::string& part_name_prefix, int32_t part_name_suffix_length, int32_t batch_size,
         int32_t shuffle_buffer_size, bool random_shuffle, bool shuffle_after_epoch, int64_t seed,
         const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
            "data_dir", "data_part_num", "part_name_prefix", "part_name_suffix_length",
            "batch_size", "shuffle_buffer_size", "random_shuffle", "shuffle_after_epoch", "seed");
        attrs.SetAllAttrs(data_dir, data_part_num, part_name_prefix, part_name_suffix_length,
                          batch_size, shuffle_buffer_size, random_shuffle, shuffle_after_epoch,
                          seed);
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor(
      "DispatchOfrecordReader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_dir, int32_t data_part_num,
         const std::string& part_name_prefix, int32_t part_name_suffix_length, int32_t batch_size,
         int32_t shuffle_buffer_size, bool random_shuffle, bool shuffle_after_epoch, int64_t seed,
         const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
            "data_dir", "data_part_num", "part_name_prefix", "part_name_suffix_length",
            "batch_size", "shuffle_buffer_size", "random_shuffle", "shuffle_after_epoch", "seed",
            "nd_sbp");
        attrs.SetAllAttrs(data_dir, data_part_num, part_name_prefix, part_name_suffix_length,
                          batch_size, shuffle_buffer_size, random_shuffle, shuffle_after_epoch,
                          seed, *JUST(GetNdSbpStrList(sbp_tuple)));
        auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<Tensor>(*op, {},
                                              OpExprInterpContext(attrs, placement, nd_sbp));
      });
  m.add_functor("DispatchOfrecordRawDecoder",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::string& name, const Shape& shape, const Symbol<DType>& data_type,
                   bool dim1_varying_length, bool truncate) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("name", "shape", "data_type",
                                                               "dim1_varying_length", "truncate");
                  attrs.SetAllAttrs(name, shape, data_type->data_type(), dim1_varying_length,
                                    truncate);
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchCoinFlip",
      [](const std::shared_ptr<OpExpr>& op, int64_t batch_size, Scalar probability, int64_t seed,
         bool has_seed, const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        auto& attrs =
            THREAD_CACHED_MUTABLE_ATTR_MAP("probability", "batch_size", "seed", "has_seed");
        attrs.SetAllAttrs(probability.As<float>(), batch_size, seed, has_seed);
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor("DispatchCoinFlip",
                [](const std::shared_ptr<OpExpr>& op, int64_t batch_size, Scalar probability,
                   int64_t seed, bool has_seed, const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("probability", "batch_size", "seed",
                                                               "has_seed", "nd_sbp");
                  attrs.SetAllAttrs(probability.As<float>(), batch_size, seed, has_seed,
                                    *JUST(GetNdSbpStrList(sbp_tuple)));
                  auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
                  return OpInterpUtil::Dispatch<Tensor>(
                      *op, {}, OpExprInterpContext(attrs, placement, nd_sbp));
                });
  m.add_functor(
      "DispatchDistributedPariticalFCSample",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& weight,
         const std::shared_ptr<Tensor>& label, const int64_t& num_sample) -> Maybe<TensorTuple> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("num_sample");
        attrs.SetAllAttrs(num_sample);
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {weight, label}, attrs);
      });
  m.add_functor(
      "DispatchCropMirrorNormalizeFromUint8",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& input, int64_t crop_h,
         int64_t crop_w, float crop_pos_x, float crop_pos_y, const std::vector<float>& mean,
         const std::vector<float>& std, const Symbol<DType>& output_dtype,
         const std::string& output_layout, const std::string& color_space) -> Maybe<Tensor> {
        auto& attrs =
            THREAD_CACHED_MUTABLE_ATTR_MAP("color_space", "output_layout", "mean", "std", "crop_h",
                                           "crop_w", "crop_pos_x", "crop_pos_y", "output_dtype");
        attrs.SetAllAttrs(color_space, output_layout, mean, std, crop_h, crop_w, crop_pos_x,
                          crop_pos_y, output_dtype->data_type());
        return OpInterpUtil::Dispatch<Tensor>(*op, input, attrs);
      });
  m.add_functor(
      "DispatchCropMirrorNormalizeFromTensorBuffer",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& input, int64_t crop_h,
         int64_t crop_w, float crop_pos_x, float crop_pos_y, const std::vector<float>& mean,
         const std::vector<float>& std, const Symbol<DType>& output_dtype,
         const std::string& output_layout, const std::string& color_space) -> Maybe<Tensor> {
        auto& attrs =
            THREAD_CACHED_MUTABLE_ATTR_MAP("color_space", "output_layout", "mean", "std", "crop_h",
                                           "crop_w", "crop_pos_x", "crop_pos_y", "output_dtype");
        attrs.SetAllAttrs(color_space, output_layout, mean, std, crop_h, crop_w, crop_pos_x,
                          crop_pos_y, output_dtype->data_type());
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
      });
  m.add_functor(
      "DispatchOfrecordImageDecoderRandomCrop",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::string& name, const std::string& color_space,
         const std::vector<float>& random_area, const std::vector<float>& random_aspect_ratio,
         int32_t num_attempts, int64_t seed, bool has_seed) -> Maybe<Tensor> {
        auto& attrs =
            THREAD_CACHED_MUTABLE_ATTR_MAP("name", "color_space", "num_attempts", "seed",
                                           "has_seed", "random_area", "random_aspect_ratio");
        attrs.SetAllAttrs(name, color_space, num_attempts, seed, has_seed, random_area,
                          random_aspect_ratio);
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
      });
  m.add_functor("DispatchOfrecordImageDecoder",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::string& name, const std::string& color_space) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("name", "color_space");
                  attrs.SetAllAttrs(name, color_space);
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor("DispatchImageDecoderRandomCropResize",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   int64_t target_width, int64_t target_height, int64_t seed, int64_t num_workers,
                   int64_t max_num_pixels, float random_area_min, float random_area_max,
                   float random_aspect_ratio_min, float random_aspect_ratio_max,
                   int64_t warmup_size, int64_t num_attempts) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
                      "target_width", "target_height", "seed", "num_workers", "max_num_pixels",
                      "random_area_min", "random_area_max", "random_aspect_ratio_min",
                      "random_aspect_ratio_max", "warmup_size", "num_attempts");
                  attrs.SetAllAttrs(target_width, target_height, seed, num_workers, max_num_pixels,
                                    random_area_min, random_area_max, random_aspect_ratio_min,
                                    random_aspect_ratio_max, warmup_size, num_attempts);
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchTensorBufferToListOfTensorsV2",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::vector<Shape>& out_shapes, const std::vector<Symbol<DType>>& out_dtypes,
         bool dynamic_out) -> Maybe<TensorTuple> {
        auto out_data_types = std::vector<DataType>();
        for (auto it = out_dtypes.begin(); it != out_dtypes.end(); it++) {
          out_data_types.emplace_back((*it)->data_type());
        }
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("out_shapes", "dynamic_out", "out_dtypes");
        attrs.SetAllAttrs(out_shapes, dynamic_out, out_data_types);
        return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, attrs);
      });
  m.add_functor("DispatchImageResizeKeepAspectRatio",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   int32_t target_size, int32_t min_size, int32_t max_size, bool resize_longer,
                   const std::string& interpolation_type) -> Maybe<TensorTuple> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
                      "target_size", "min_size", "max_size", "resize_longer", "interpolation_type");
                  attrs.SetAllAttrs(target_size, min_size, max_size, resize_longer,
                                    interpolation_type);
                  return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, attrs);
                });
  m.add_functor("DispatchImageResizeToFixed",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   int64_t target_width, int64_t target_height, int64_t channels,
                   const Symbol<DType>& data_type,
                   const std::string& interpolation_type) -> Maybe<TensorTuple> {
                  auto& attrs =
                      THREAD_CACHED_MUTABLE_ATTR_MAP("target_width", "target_height", "channels",
                                                     "data_type", "interpolation_type");
                  attrs.SetAllAttrs(target_width, target_height, channels, data_type->data_type(),
                                    interpolation_type);
                  return OpInterpUtil::Dispatch<TensorTuple>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchImageDecode",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
         const std::string& color_space, const Symbol<DType>& data_type) -> Maybe<Tensor> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("color_space", "data_type");
        attrs.SetAllAttrs(color_space, data_type->data_type());
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
      });
  m.add_functor("DispatchImageNormalize",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::vector<float>& mean, const std::vector<float>& std) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("std", "mean");
                  attrs.SetAllAttrs(std, mean);
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor("DispatchCOCOReader",
                [](const std::shared_ptr<OpExpr>& op, const std::string& image_dir,
                   const std::string& annotation_file, int64_t batch_size, bool shuffle_after_epoch,
                   int64_t random_seed, bool group_by_ratio, bool remove_images_without_annotations,
                   bool stride_partition, int64_t session_id,
                   const Optional<Symbol<Device>>& device) -> Maybe<TensorTuple> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
                      "session_id", "annotation_file", "image_dir", "batch_size",
                      "shuffle_after_epoch", "random_seed", "group_by_ratio",
                      "remove_images_without_annotations", "stride_partition");
                  attrs.SetAllAttrs(session_id, annotation_file, image_dir, batch_size,
                                    shuffle_after_epoch, random_seed, group_by_ratio,
                                    remove_images_without_annotations, stride_partition);
                  return OpInterpUtil::Dispatch<TensorTuple>(
                      *op, {}, OpExprInterpContext(attrs, JUST(device)));
                });
  m.add_functor("DispatchCOCOReader",
                [](const std::shared_ptr<OpExpr>& op, const std::string& image_dir,
                   const std::string& annotation_file, int64_t batch_size, bool shuffle_after_epoch,
                   int64_t random_seed, bool group_by_ratio, bool remove_images_without_annotations,
                   bool stride_partition, int64_t session_id, const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<TensorTuple> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
                      "session_id", "annotation_file", "image_dir", "batch_size",
                      "shuffle_after_epoch", "random_seed", "group_by_ratio",
                      "remove_images_without_annotations", "stride_partition", "nd_sbp");
                  attrs.SetAllAttrs(session_id, annotation_file, image_dir, batch_size,
                                    shuffle_after_epoch, random_seed, group_by_ratio,
                                    remove_images_without_annotations, stride_partition,
                                    *JUST(GetNdSbpStrList(sbp_tuple)));
                  auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
                  return OpInterpUtil::Dispatch<TensorTuple>(
                      *op, {}, OpExprInterpContext(attrs, placement, nd_sbp));
                });
  m.add_functor(
      "DispatchImageBatchAlign",
      [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input, int32_t alignment,
         const Shape& shape, const Symbol<DType>& data_type, bool dynamic_out) -> Maybe<Tensor> {
        auto& attrs =
            THREAD_CACHED_MUTABLE_ATTR_MAP("shape", "data_type", "alignment", "dynamic_out");
        attrs.SetAllAttrs(shape, data_type->data_type(), alignment, dynamic_out);
        return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
      });
  m.add_functor("DispatchOfrecordBytesDecoder",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::string& name) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("name");
                  attrs.SetAllAttrs(name);
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchMegatronGptMmapDataLoader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_file_prefix, int64_t seq_length,
         int64_t label_length, int64_t num_samples, int64_t batch_size, const Symbol<DType>& dtype,
         const std::vector<int64_t>& split_sizes, int64_t split_index, bool shuffle,
         int64_t random_seed, const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
            "data_file_prefix", "seq_length", "label_length", "num_samples", "batch_size", "dtype",
            "split_sizes", "split_index", "shuffle", "random_seed");
        attrs.SetAllAttrs(data_file_prefix, seq_length, label_length, num_samples, batch_size,
                          dtype->data_type(), split_sizes, split_index, shuffle, random_seed);
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor(
      "DispatchMegatronGptMmapDataLoader",
      [](const std::shared_ptr<OpExpr>& op, const std::string& data_file_prefix, int64_t seq_length,
         int64_t label_length, int64_t num_samples, int64_t batch_size, const Symbol<DType>& dtype,
         const std::vector<int64_t>& split_sizes, int64_t split_index, bool shuffle,
         int64_t random_seed, const Symbol<ParallelDesc>& placement,
         const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
            "data_file_prefix", "seq_length", "label_length", "num_samples", "batch_size", "dtype",
            "split_sizes", "split_index", "shuffle", "random_seed");
        attrs.SetAllAttrs(data_file_prefix, seq_length, label_length, num_samples, batch_size,
                          dtype->data_type(), split_sizes, split_index, shuffle, random_seed);
        auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
        return OpInterpUtil::Dispatch<Tensor>(*op, {},
                                              OpExprInterpContext(attrs, placement, nd_sbp));
      });
  m.add_functor("DispatchRmspropUpdate",
                [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs,
                   float learning_rate, double scale, float l1, float l2, bool centered,
                   float epsilon, float decay_rate, float weight_decay) -> Maybe<void> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "l1",
                                                               "l2", "centered", "epsilon",
                                                               "decay_rate", "weight_decay");
                  attrs.SetAllAttrs(learning_rate, scale, l1, l2, centered, epsilon, decay_rate,
                                    weight_decay);
                  JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
                  return Maybe<void>::Ok();
                });
  m.add_functor(
      "DispatchAdamUpdate",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs, float learning_rate,
         float bias_correction1, float bias_correction2, double scale, float l1, float l2,
         float beta1, float beta2, float epsilon, float weight_decay, bool amsgrad,
         bool do_bias_correction) -> Maybe<void> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
            "learning_rate_val", "bias_correction1_val", "bias_correction2_val", "scale", "l1",
            "l2", "beta1", "beta2", "epsilon", "weight_decay", "amsgrad", "do_bias_correction");
        attrs.SetAllAttrs(learning_rate, bias_correction1, bias_correction2, scale, l1, l2, beta1,
                          beta2, epsilon, weight_decay, amsgrad, do_bias_correction);
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
        return Maybe<void>::Ok();
      });
  m.add_functor("DispatchAdagradUpdate",
                [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs,
                   float learning_rate, double scale, float l1, float l2, float lr_decay,
                   float weight_decay, float epsilon, int32_t train_step) -> Maybe<void> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "l1",
                                                               "l2", "lr_decay", "weight_decay",
                                                               "epsilon", "train_step_val");
                  attrs.SetAllAttrs(learning_rate, scale, l1, l2, lr_decay, weight_decay, epsilon,
                                    train_step);
                  JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
                  return Maybe<void>::Ok();
                });
  m.add_functor(
      "DispatchMomentumUpdate",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs, float learning_rate,
         double scale, float l1, float l2, float beta, float dampening, bool nesterov,
         bool maximize, float weight_decay) -> Maybe<void> {
        auto& attrs =
            THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "l1", "l2", "beta",
                                           "dampening", "nesterov", "maximize", "weight_decay");
        attrs.SetAllAttrs(learning_rate, scale, l1, l2, beta, dampening, nesterov, maximize,
                          weight_decay);
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
        return Maybe<void>::Ok();
      });
  m.add_functor(
      "DispatchSgdUpdate",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs, float learning_rate,
         double scale, float l1, float l2, float weight_decay) -> Maybe<void> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "l1", "l2",
                                                     "weight_decay");
        attrs.SetAllAttrs(learning_rate, scale, l1, l2, weight_decay);
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
        return Maybe<void>::Ok();
      });
  m.add_functor("DispatchLambUpdate",
                [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs,
                   float learning_rate, float bias_correction1, float bias_correction2,
                   double scale, float l1, float l2, float beta1, float beta2, float epsilon,
                   float weight_decay, bool do_bias_correction) -> Maybe<void> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
                      "learning_rate_val", "bias_correction1_val", "bias_correction2_val", "scale",
                      "l1", "l2", "beta1", "beta2", "epsilon", "weight_decay",
                      "do_bias_correction");
                  attrs.SetAllAttrs(learning_rate, bias_correction1, bias_correction2, scale, l1,
                                    l2, beta1, beta2, epsilon, weight_decay, do_bias_correction);
                  JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
                  return Maybe<void>::Ok();
                });
  m.add_functor("DispatchFtrlUpdate",
                [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs,
                   float learning_rate, double scale, float l1, float l2, float lr_power,
                   float lambda1, float lambda2, float beta, float weight_decay) -> Maybe<void> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "l1",
                                                               "l2", "lr_power", "lambda1",
                                                               "lambda2", "beta", "weight_decay");
                  attrs.SetAllAttrs(learning_rate, scale, l1, l2, lr_power, lambda1, lambda2, beta,
                                    weight_decay);
                  JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
                  return Maybe<void>::Ok();
                });
  m.add_functor(
      "DispatchAdadeltaUpdate",
      [](const std::shared_ptr<OpExpr>& op, const TensorTuple& inputs, float learning_rate,
         double scale, float l1, float l2, float rho, float epsilon, bool maximize,
         float weight_decay) -> Maybe<void> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("learning_rate_val", "scale", "l1", "l2",
                                                     "rho", "epsilon", "maximize", "weight_decay");
        attrs.SetAllAttrs(learning_rate, scale, l1, l2, rho, epsilon, maximize, weight_decay);
        JUST(OpInterpUtil::Dispatch<TensorTuple>(*op, inputs, attrs));
        return Maybe<void>::Ok();
      });
  m.add_functor("DispatchEagerCclAllReduce",
                [](const std::shared_ptr<OpExpr>& op, const std::shared_ptr<Tensor>& input,
                   const std::string& parallel_conf) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("parallel_conf");
                  attrs.SetAllAttrs(parallel_conf);
                  return OpInterpUtil::Dispatch<Tensor>(*op, {input}, attrs);
                });
  m.add_functor(
      "DispatchRawReader",
      [](const std::shared_ptr<OpExpr>& op, const std::vector<std::string>& files,
         const Shape& shape, const Symbol<DType>& data_type, const int64_t batch_size,
         const bool random_shuffle, const int64_t shuffle_block_size, int64_t random_seed,
         const Optional<Symbol<Device>>& device) -> Maybe<Tensor> {
        auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("files", "shape", "data_type", "batch_size",
                                                     "random_shuffle", "shuffle_block_size", "seed",
                                                     "nd_sbp");
        attrs.SetAllAttrs(files, shape, data_type->data_type(), batch_size, random_shuffle,
                          shuffle_block_size, random_seed, std::vector<std::string>());
        return OpInterpUtil::Dispatch<Tensor>(*op, {}, OpExprInterpContext(attrs, JUST(device)));
      });
  m.add_functor("DispatchRawReader",
                [](const std::shared_ptr<OpExpr>& op, const std::vector<std::string>& files,
                   const Shape& shape, const Symbol<DType>& data_type, const int64_t batch_size,
                   const bool random_shuffle, const int64_t shuffle_block_size, int64_t random_seed,
                   const Symbol<ParallelDesc>& placement,
                   const std::vector<Symbol<SbpParallel>>& sbp_tuple) -> Maybe<Tensor> {
                  auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP(
                      "files", "shape", "data_type", "batch_size", "random_shuffle",
                      "shuffle_block_size", "seed", "nd_sbp");
                  attrs.SetAllAttrs(files, shape, data_type->data_type(), batch_size,
                                    random_shuffle, shuffle_block_size, random_seed,
                                    *JUST(GetNdSbpStrList(sbp_tuple)));
                  auto nd_sbp = JUST(GetNdSbp(sbp_tuple));
                  return OpInterpUtil::Dispatch<Tensor>(
                      *op, {}, OpExprInterpContext(attrs, placement, nd_sbp));
                });
  m.add_functor<impl::OneEmbeddingIdShuffleFunctor>("OneEmbeddingIdShuffle");
  m.add_functor<impl::OneEmbeddingEmbeddingShuffleFunctor>("OneEmbeddingEmbeddingShuffle");
  m.add_functor<impl::OneEmbeddingEmbeddingGradientShuffleFunctor>(
      "OneEmbeddingEmbeddingGradientShuffle");
  m.add_functor<impl::OneEmbeddingLookupFunctor>("OneEmbeddingLookup");
  m.add_functor<impl::OneEmbeddingFusedLookupFunctor>("OneEmbeddingFusedLookup");
  m.add_functor<impl::OneEmbeddingEmbeddingPutFunctor>("OneEmbeddingEmbeddingPut");
  m.add_functor<impl::OneEmbeddingUniqueKeyValuePairFunctor>("OneEmbeddingUniqueKeyValuePair");
  m.add_functor<impl::OneEmbeddingSgdUpdateFunctor>("OneEmbeddingSgdUpdate");
  m.add_functor<impl::OneEmbeddingAdamUpdateFunctor>("OneEmbeddingAdamUpdate");
  m.add_functor<impl::OneEmbeddingAdagradUpdateFunctor>("OneEmbeddingAdagradUpdate");
  m.add_functor<impl::OneEmbeddingFtrlUpdateFunctor>("OneEmbeddingFtrlUpdate");
}

}  // namespace impl

}  // namespace functional
}  // namespace one
}  // namespace oneflow
