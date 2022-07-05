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
#include "oneflow/core/job_rewriter/auto_mixed_precision_lists.h"

namespace oneflow {

const AMPList& AutoMixedPrecisionLists::WhiteList() {
  static AMPList white_list = {"matmul",
                               "batch_matmul",
                               "conv2d",
                               "amp_white_identity",
                               "broadcast_matmul",
                               "fused_self_attention_query_mul_key_and_value",
                               "prelu",
                               "tf_prelu",
                               "cublas_fused_mlp",
                               "fused_matmul_bias_add_relu_dropout",
                               "fused_dot_feature_interaction",
                               "embedding_lookup_placeholder",
                               "binary_cross_entropy_with_logits_reduce_mean"};
  return white_list;
}

const AMPList& AutoMixedPrecisionLists::BlackList() {
  // TODO(niuchong): reduce_mean?
  static AMPList black_list = {};
  return black_list;
}

const AMPList& AutoMixedPrecisionLists::GrayList() {
  static AMPList gray_list = {"add_n",
                              "tf_avg_pool_1d",
                              "tf_avg_pool_2d",
                              "tf_avg_pool_3d",
                              "bias_add",
                              "sigmoid_v2",
                              "tanh",
                              "sqrt",
                              "scalar_mul",
                              "scalar_add",
                              "scalar_div",
                              "broadcast_add",
                              "broadcast_sub",
                              "broadcast_mul",
                              "broadcast_div",
                              "layer_norm",
                              "dropout",
                              "softmax",
                              "log_softmax",
                              "gelu",
                              "normalization",
                              "normalization_add_relu",
                              "sparse_softmax_cross_entropy",
                              "sparse_softmax_cross_entropy_ms",
                              "nll",
                              "fused_tril_scale_softmax_mask_scale",
                              "fused_scale_mask_softmax_dropout",
                              "fused_scale_mask_softmax",
                              "fused_bias_add_gelu",
                              "fused_bias_add_mask_scale",
                              "acc"};
  return gray_list;
}

const AMPList& AutoMixedPrecisionLists::ClearList() {
  // TODO(niuchong): tuple_identity
  static AMPList clear_list = {"gather",
                               "tf_max_pool_1d",
                               "tf_max_pool_2d",
                               "tf_max_pool_3d",
                               "reshape",
                               "relu",
                               "transpose",
                               "random_mask_like",
                               "concat",
                               "pad",
                               "same_padding",
                               "tril",
                               "slice",
                               "fused_scale_tril",
                               "identity",
                               "flatten",
                               "squeeze",
                               "embedding",
                               "expand_dims",
                               "cast_to_static_shape",
                               "parallel_cast",
                               "hierarchical_parallel_cast",
                               "hierarchical_parallel_cast_like",
                               "repeat",
                               "unpack",
                               "pack",
                               "nvtx_start",
                               "nvtx_end",
                               "narrow"};

  return clear_list;
}

}  // namespace oneflow
