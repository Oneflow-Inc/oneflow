#include "oneflow/core/job_completer/auto_mixed_precision_lists.h"

namespace oneflow {

const AMPList& AutoMixedPrecisionLists::WhiteList() {
  static AMPList white_list = {"matmul", "batch_matmul", "conv2d"};
  return white_list;
}

const AMPList& AutoMixedPrecisionLists::BlackList() {
  // TODO(niuchong): reduce_mean?
  static AMPList black_list = {};
  return black_list;
}

const AMPList& AutoMixedPrecisionLists::GrayList() {
  static AMPList gray_list = {"add_n",         "avg_pool_1d",   "avg_pool_2d",   "avg_pool_3d",
                              "bias_add",      "multiply",      "sigmoid",       "tanh",
                              "sqrt",          "scalar_mul",    "scalar_add",    "broadcast_add",
                              "broadcast_sub", "broadcast_mul", "broadcast_div", "layer_norm",
                              "dropout",       "softmax",       "gelu",          "normalization"};
  return gray_list;
}

const AMPList& AutoMixedPrecisionLists::ClearList() {
  // TODO(niuchong): identity, tuple_identity, keep_header_only?
  static AMPList clear_list = {"gather",      "max_pool_1d",      "max_pool_2d",
                               "max_pool_3d", "reshape",          "relu",
                               "transpose",   "random_mask_like", "concat"};

  return clear_list;
}

}  // namespace oneflow
