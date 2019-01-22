#include "oneflow/core/operator/bias_add_op.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

void BiasAddOp::InitFromOpConf() {
  CHECK(op_conf().has_bias_add_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
  EnrollConstBufBn("bias_multiplier");
}

const PbMessage& BiasAddOp::GetCustomizedConf() const { return op_conf().bias_add_conf(); }

bool BiasAddOp::IsInputBnInOpAllowedModelSplit(const std::string& ibn) const {
  CHECK(std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end());
  return ibn == "b";
}

void BiasAddOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                               const ParallelContext* parallel_ctx) const {
  const BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  const BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");

  CHECK_EQ(a_blob_desc->shape().NumAxes(), 2);
  CHECK_EQ(b_blob_desc->shape().NumAxes(), 1);
  CHECK_EQ(a_blob_desc->shape().At(1), b_blob_desc->shape().At(0));
  CHECK_EQ(a_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(b_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());

  *GetBlobDesc4BnInOp("out") = *a_blob_desc;
  GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() = Shape({a_blob_desc->shape().At(0), 1});
}

void BiasAddOp::InferOutputBlobModelSplitAxis(
    std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
    std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
    const ParallelContext* parallel_context) const {
  if (parallel_context->policy() == kDataParallel) {
    *ModelSplitAxis4BnInOp("out") = -1;
  } else if (parallel_context->policy() == kModelParallel) {
    *ModelSplitAxis4BnInOp("out") = 1;
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_OP(OperatorConf::kBiasAddConf, BiasAddOp);

}  // namespace oneflow
