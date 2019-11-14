#include "oneflow/core/operator/one_hot_op.h"

namespace oneflow {

void OneHotOp::InitFromOpConf() {
  CHECK(op_conf().has_one_hot_conf());
  EnrollInputBn("indices", false);
  EnrollOutputBn("out", false);
}

const PbMessage& OneHotOp::GetCustomizedConf() const { return op_conf().one_hot_conf(); }

Maybe<void> OneHotOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const OneHotOpConf& conf = op_conf().one_hot_conf();
  const int64_t depth = conf.depth();
  const DataType data_type = conf.has_data_type() ? conf.data_type() : job_desc().DefaultDataType();
  CHECK_GT_OR_RETURN(depth, 0);
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK_OR_RETURN(IsIntegralDataType(indices->data_type()));
  CHECK_GT_OR_RETURN(indices->shape().NumAxes(), 0);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *indices;
  out->set_data_type(data_type);
  std::vector<int64_t> dim_vec = indices->shape().dim_vec();
  dim_vec.push_back(depth);
  out->mut_shape() = Shape(dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> OneHotOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("indices", 0)
      .Split("out", 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kOneHotConf, OneHotOp);

}  // namespace oneflow
