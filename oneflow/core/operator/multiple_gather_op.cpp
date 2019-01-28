#include "oneflow/core/operator/multiple_gather_op.h"

namespace oneflow {

void MultipleGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_multiple_gather_conf());
  EnrollRepeatedInputBn("indices", false);
  EnrollInputBn("in");
  EnrollRepeatedOutputBn("out");
}

const PbMessage& MultipleGatherOp::GetCustomizedConf() const {
  return op_conf().multiple_gather_conf();
}

void MultipleGatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const MultipleGatherOpConf& conf = op_conf().multiple_gather_conf();
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  CHECK_GT(conf.indices().size(), 0);
  CHECK_EQ(conf.indices().size(), conf.out().size());
  FOR_RANGE(int32_t, i, 0, conf.indices().size()) {
    const BlobDesc* indices = GetBlobDesc4BnInOp(GenRepeatedBn("indices", i));
    CHECK(IsIntegralDataType(indices->data_type()));
    CHECK_GT(indices->shape().NumAxes(), 0);
    BlobDesc* out = GetBlobDesc4BnInOp(GenRepeatedBn("out", i));
    *out = *in;
    std::vector<int64_t> dim_vec;
    dim_vec.insert(dim_vec.end(), indices->shape().dim_vec().cbegin(),
                   indices->shape().dim_vec().cend());
    dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin() + 1, in->shape().dim_vec().end());
    out->mut_shape() = Shape(dim_vec);
  }
}

bool MultipleGatherOp::IsInputBlobAllowedModelSplit(const std::string& ibn) const {
  CHECK(std::find(input_bns().begin(), input_bns().end(), ibn) != input_bns().end());
  return ibn == "in";
}

void MultipleGatherOp::InferOutputBlobModelSplitAxis(
    std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
    std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
    const ParallelContext* parallel_context) const {
  CHECK_EQ(parallel_context->policy(), kDataParallel);
  NaiveInferOutputBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
}

REGISTER_OP(OperatorConf::kMultipleGatherConf, MultipleGatherOp);

}  // namespace oneflow
