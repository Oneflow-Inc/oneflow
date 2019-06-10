#include "oneflow/core/job_completer/auto_mixed_precision_lists.h"

namespace oneflow {

const HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash>& AutoMixedPrecisionLists::WhiteList() {
  static HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash> white_list = {OperatorConf::kMatmulConf,
                                                                         OperatorConf::kConv2DConf};
  return white_list;
}

const HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash>& AutoMixedPrecisionLists::BlackList() {
  static HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash> black_list = {
      OperatorConf::kMeanConf, OperatorConf::kSoftmaxConf};
  return black_list;
}

const HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash>& AutoMixedPrecisionLists::GrayList() {
  static HashSet<OperatorConf::OpTypeCase, OpTypeCaseHash> gray_list = {
      OperatorConf::kAddConf,          OperatorConf::kAveragePooling1DConf,
      OperatorConf::kMaxPooling1DConf, OperatorConf::kAveragePooling2DConf,
      OperatorConf::kMaxPooling2DConf, OperatorConf::kAveragePooling3DConf,
      OperatorConf::kMaxPooling3DConf, OperatorConf::kBiasAddConf,
      OperatorConf::kMultiplyConf,     OperatorConf::kSigmoidConf,
      OperatorConf::kTanhConf,         OperatorConf::kSqrtConf,
      OperatorConf::kScalarMulConf,    OperatorConf::kScalarAddConf,
      OperatorConf::kBroadcastAddConf, OperatorConf::kBroadcastSubConf,
      OperatorConf::kBroadcastMulConf, OperatorConf::kBroadcastDivConf};
  return gray_list;
}

}  // namespace oneflow
