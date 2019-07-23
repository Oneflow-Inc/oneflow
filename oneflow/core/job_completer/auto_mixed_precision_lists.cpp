#include "oneflow/core/job_completer/auto_mixed_precision_lists.h"

namespace oneflow {

const AMPList& AutoMixedPrecisionLists::WhiteList() {
  static AMPList white_list = {OperatorConf::kMatmulConf, OperatorConf::kConv2DConf};
  return white_list;
}

const AMPList& AutoMixedPrecisionLists::BlackList() {
  static AMPList black_list = {OperatorConf::kMeanConf, OperatorConf::kSoftmaxConf};
  return black_list;
}

const AMPList& AutoMixedPrecisionLists::GrayList() {
  static AMPList gray_list = {OperatorConf::kAddConf,
                              OperatorConf::kAveragePooling1DConf,
                              OperatorConf::kAveragePooling2DConf,
                              OperatorConf::kAveragePooling3DConf,
                              OperatorConf::kBiasAddConf,
                              OperatorConf::kMultiplyConf,
                              OperatorConf::kSigmoidConf,
                              OperatorConf::kTanhConf,
                              OperatorConf::kSqrtConf,
                              OperatorConf::kScalarMulConf,
                              OperatorConf::kScalarAddConf,
                              OperatorConf::kBroadcastAddConf,
                              OperatorConf::kBroadcastSubConf,
                              OperatorConf::kBroadcastMulConf,
                              OperatorConf::kBroadcastDivConf,
                              OperatorConf::kLayerNormConf};
  return gray_list;
}

const AMPList& AutoMixedPrecisionLists::ClearList() {
  static AMPList gray_list = {OperatorConf::kGatherConf,        OperatorConf::kIdentityConf,
                              OperatorConf::kTupleIdentityConf, OperatorConf::kMaxPooling1DConf,
                              OperatorConf::kMaxPooling2DConf,  OperatorConf::kMaxPooling3DConf,
                              OperatorConf::kReshapeConf,       OperatorConf::kReluConf,
                              OperatorConf::kTransposeConf};
  return gray_list;
}

}  // namespace oneflow
