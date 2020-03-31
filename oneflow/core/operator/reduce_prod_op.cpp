#include "oneflow/core/operator/reduce_prod_op.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

void ReduceProdOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_prod_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollTmpBn("fw_tmp");
}

const PbMessage& ReduceProdOp::GetCustomizedConf() const { return op_conf().reduce_prod_conf(); }

Maybe<void> ReduceProdOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const ReduceProdOpConf& conf = op_conf().reduce_prod_conf();
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob->set_data_type(in_blob->data_type());
  if (conf.axis().empty()) {
    if (conf.keep_dims()) {
      out_blob->mut_shape() = Shape::Ones(in_blob->shape().NumAxes());
    } else {
      out_blob->mut_shape() = Shape({1});
    }
  } else {
    const AxisVector axis_vec = {conf.axis().begin(), conf.axis().end()};
    const Shape& reduced_shape = CreateReducedShape(in_blob->shape(), axis_vec);
    if (conf.keep_dims()) {
      out_blob->mut_shape() = reduced_shape;
    } else {
      out_blob->mut_shape() = reduced_shape.RemoveOnes(axis_vec);
    }
  }
  out_blob->set_is_dynamic(in_blob->is_dynamic());
  return Maybe<void>::Ok();
}

Maybe<void> ReduceProdOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  const auto& reduced_axes = op_conf().reduce_prod_conf().axis();
  const bool keep_dims = op_conf().reduce_prod_conf().keep_dims();
  HashSet<int64_t> conf_axes = {reduced_axes.begin(), reduced_axes.end()};
  if (BatchAxis4BnInOp("in")->has_value() && !conf_axes.empty()
      && conf_axes.find(BatchAxis4BnInOp("in")->value()) == conf_axes.end()) {
    *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
  } else if (conf_axes.empty() && keep_dims == false) {
    BatchAxis4BnInOp("out")->set_value(0);
  } else {
    BatchAxis4BnInOp("out")->clear_value();
  }
  return Maybe<void>::Ok();
}

Maybe<void> ReduceProdOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int32_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
  auto IsReducedAxis =
      ReduceSbpUtil::MakePredicatorIsReducedAxis(op_conf().reduce_prod_conf().axis(), num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
    } else {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .Split(output_bns(), i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReduceProdConf, ReduceProdOp);

}  // namespace oneflow