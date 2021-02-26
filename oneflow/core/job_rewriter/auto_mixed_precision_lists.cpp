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
  static AMPList white_list = {"matmul", "batch_matmul", "conv2d", "amp_white_identity"};
  return white_list;
}

const AMPList& AutoMixedPrecisionLists::BlackList() {
  // TODO(niuchong): reduce_mean?
  static AMPList black_list = {};
  return black_list;
}

const AMPList& AutoMixedPrecisionLists::GrayList() {
  static AMPList gray_list = {"add_n",         "avg_pool_1d",   "avg_pool_2d",
                              "avg_pool_3d",   "bias_add",      "multiply",
                              "sigmoid",       "tanh",          "sqrt",
                              "scalar_mul",    "scalar_add",    "broadcast_add",
                              "broadcast_sub", "broadcast_mul", "broadcast_div",
                              "layer_norm",    "dropout",       "softmax",
                              "gelu",          "normalization", "normalization_add_relu"};
  return gray_list;
}

const AMPList& AutoMixedPrecisionLists::ClearList() {
  // TODO(niuchong): identity, tuple_identity?
  static AMPList clear_list = {"gather",
                               "max_pool_1d",
                               "max_pool_2d",
                               "max_pool_3d",
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
                               "expand_dims",
                               "cast_to_static_shape",
                               "parallel_cast"};

  return clear_list;
}

}  // namespace oneflow
