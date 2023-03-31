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
                               "conv_data_grad",
                               "conv_filter_grad",
                               "conv_bias_grad",
                               "amp_white_identity",
                               "broadcast_matmul",
                               "broadcast_matmul_grad_b",
                               "fused_self_attention_query_mul_key_and_value",
                               "fused_self_attention_query_mul_key_and_value_grad",
                               "prelu",
                               "prelu_grad",
                               "tf_prelu",
                               "tf_prelu_grad",
                               "cublas_fused_mlp",
                               "cublas_fused_mlp_grad",
                               "fused_matmul_bias",
                               "cublas_bias_add_relu_matmul_grad",
                               "fused_glu",
                               "fused_glu_without_linear_grad",
                               "fused_matmul_bias_add_relu_dropout",
                               "fused_relu_dropout_grad",
                               "fused_dot_feature_interaction",
                               "fused_dot_feature_interaction_grad",
                               "one_embedding_fused_lookup",
                               "one_embedding_fused_lookup_grad",
                               "binary_cross_entropy_with_logits_reduce_mean",
                               "binary_cross_entropy_with_logits_reduce_mean_grad",
                               "fused_cross_feature_interaction",
                               "fused_cross_feature_interaction_v1_grad",
                               "fused_cross_feature_interaction_v2_grad",
                               "fused_multi_head_attention_inference",
                               "grouped_matmul_bias"};
  return white_list;
}

const AMPList& AutoMixedPrecisionLists::BlackList() {
  // TODO(niuchong): reduce_mean?
  static AMPList black_list = {"amp_black_identity"};
  return black_list;
}

const AMPList& AutoMixedPrecisionLists::GrayList() {
  static AMPList gray_list = {"add_n",
                              "tf_avg_pool_1d",
                              "tf_avg_pool_1d_grad",
                              "tf_avg_pool_2d",
                              "tf_avg_pool_2d_grad",
                              "tf_avg_pool_3d",
                              "tf_avg_pool_3d_grad",
                              "avg_pool_1d",
                              "avg_pool_1d_grad",
                              "avg_pool_2d",
                              "avg_pool_2d_grad",
                              "avg_pool_3d",
                              "avg_pool_3d_grad",
                              "bias_add",
                              "reduce_sum",
                              "reduce_sum_like",
                              "sigmoid_grad",
                              "tanh",
                              "tanh_grad",
                              "sqrt",
                              "sqrt_grad",
                              "scalar_mul",
                              "scalar_mul_by_tensor",
                              "scalar_add",
                              "scalar_div",
                              "scalar_pow",
                              "broadcast_add",
                              "broadcast_sub",
                              "broadcast_mul",
                              "broadcast_div",
                              "layer_norm",
                              "layer_norm_param_grad",
                              "layer_norm_grad",
                              "skip_layer_norm",
                              "rms_norm",
                              "rms_norm_grad",
                              "rms_norm_param_grad",
                              "dropout",
                              "dropout_grad",
                              "softmax",
                              "softmax_grad",
                              "log_softmax",
                              "log_softmax_grad",
                              "gelu",
                              "gelu_grad",
                              "fast_gelu",
                              "fast_gelu_grad",
                              "normalization",
                              "normalization_grad",
                              "normalization_add_relu",
                              "normalization_add_relu_grad",
                              "sparse_softmax_cross_entropy",
                              "sparse_softmax_cross_entropy_grad",
                              "nll",
                              "nll_grad",
                              "fused_tril_scale_softmax_mask_scale",
                              "fused_tril_scale_softmax_mask_scale_grad",
                              "fused_scale_mask_softmax_dropout",
                              "fused_scale_mask_softmax_dropout_grad",
                              "fused_scale_mask_softmax",
                              "fused_scale_mask_softmax_grad",
                              "fused_bias_add_scale_mask_softmax_dropout",
                              "fused_bias_add_gelu",
                              "fused_bias_add_gelu_grad",
                              "fused_bias_add_mask_scale",
                              "fused_fast_gelu_mul",
                              "fused_fast_gelu_mul_grad",
                              "acc",
                              "reciprocal",
                              "reciprocal_no_nan",
                              "group_norm",
                              "group_norm_param_grad",
                              "group_norm_grad",
                              "silu",
                              "silu_grad",
                              "fused_weighted_sum"};
  return gray_list;
}

const AMPList& AutoMixedPrecisionLists::ClearList() {
  // TODO(niuchong): tuple_identity
  static AMPList clear_list = {"broadcast_like",
                               "gather",
                               "gather_nd",
                               "scatter_nd",
                               "scatter_nd_like",
                               "unsorted_segment_sum_like",
                               "tf_max_pool_1d",
                               "tf_max_pool_1d_grad",
                               "tf_max_pool_2d",
                               "tf_max_pool_2d_grad",
                               "tf_max_pool_3d",
                               "tf_max_pool_3d_grad",
                               "max_pool_1d",
                               "max_pool_1d_grad",
                               "max_pool_2d",
                               "max_pool_2d_grad",
                               "max_pool_3d",
                               "max_pool_3d_grad",
                               "reshape",
                               "reshape_like",
                               "relu",
                               "relu_grad",
                               "transpose",
                               "random_mask_like",
                               "concat",
                               "split_like",
                               "pad",
                               "same_padding",
                               "same_padding_grad",
                               "tril",
                               "slice",
                               "slice_grad",
                               "fused_scale_tril",
                               "identity",
                               "squeeze",
                               "embedding",
                               "embedding_grad",
                               "expand",
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
                               "narrow",
                               "narrow_grad",
                               "ones_like",
                               "pinned_identity",
                               "to_contiguous",
                               "copy",
                               "where",
                               "upsample_nearest_2d",
                               "fill_"};

  return clear_list;
}

}  // namespace oneflow
