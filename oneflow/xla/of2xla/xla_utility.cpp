#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/xla/of2xla/xla_utility.h"

namespace oneflow {

#define OP_TYPE_CASE(op)  OperatorConf::k##op##Conf

static std::unordered_map<int32_t, std::string> op_type2string_map = {
  {OP_TYPE_CASE(Matmul),  "MatMul"},
  {OP_TYPE_CASE(Relu), "Relu"},
  {OP_TYPE_CASE(FullyConnected), "FullyConnected"},
  {OP_TYPE_CASE(BiasAdd), "BiasAdd"},
  {OP_TYPE_CASE(Reshape), "Reshape"},
  {OP_TYPE_CASE(Identity), "Identity"},
  {OP_TYPE_CASE(ReshapeLike), "ReshapeLike"},
  {OP_TYPE_CASE(Cast), "Cast"},
  {OP_TYPE_CASE(ScalarAdd), "ScalarAdd"},
  {OP_TYPE_CASE(ScalarMul), "ScalarMul"},
  {OP_TYPE_CASE(Transpose), "Transpose"},
  {OP_TYPE_CASE(BroadcastAdd), "BcastAdd"},
  {OP_TYPE_CASE(BroadcastMul), "BcastMul"},
  {OP_TYPE_CASE(BroadcastDiv), "BcastDiv"},
  {OP_TYPE_CASE(Add), "Add"},
  {OP_TYPE_CASE(Sigmoid), "Sigmoid"},
  {OP_TYPE_CASE(Tanh), "Tanh"},
  {OP_TYPE_CASE(TanhGrad), "TanhGrad"},
  {OP_TYPE_CASE(Gelu), "Gelu"},
  {OP_TYPE_CASE(GeluGrad), "GeluGrad"},
  {OP_TYPE_CASE(Gather), "Gather"},
  {OP_TYPE_CASE(BatchGather), "BatchGather"},
  {OP_TYPE_CASE(Softmax), "Softmax"},
  {OP_TYPE_CASE(SoftmaxGrad), "SoftmaxGrad"},
  {OP_TYPE_CASE(LayerNorm), "LayerNorm"},
  {OP_TYPE_CASE(LayerNormParamGrad), "LayerNormParamGrad"},
  {OP_TYPE_CASE(LayerNormGrad), "LayerNormGrad"},
  {OP_TYPE_CASE(ReduceSum), "ReduceSum"},
  {OP_TYPE_CASE(AdamModelUpdate), "AdamModelUpdate"},
  {OP_TYPE_CASE(AdamOptimizer), "AdamOptimizer"},
  {OP_TYPE_CASE(ClipGradient), "ClipGradient"},
  // TODO(hjchen2)
};

std::string ExtractOpTypeAsString(const OperatorConf &conf) {
  const auto it = op_type2string_map.find(conf.op_type_case());
  if (it != op_type2string_map.end()) {
    return it->second;
  } else {
    // Return empty if the operator is not in the translation map
    return "";
  }
}

std::string BlobName(const LogicalBlobId &lbi) {
  CHECK_EQ(lbi.has_op_name(), true);
  CHECK_EQ(lbi.has_blob_name(), true);
  if (lbi.op_name() == "") {
    return lbi.blob_name();
  }
  return GenLogicalBlobName(lbi);
}

LogicalBlobId BlobId(const std::string &blob_name) {
  size_t pos = blob_name.find('/');
  if (pos == std::string::npos) {
    return GenLogicalBlobId("/" + blob_name);
  } else {
    return GenLogicalBlobId(blob_name);
  }
}

}  // namespace oneflow
