#include "oneflow/core/operator/reduce_sum_op.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReduceSumOp::InitFromOpConf() {
  CHECK(op_conf().has_reduce_sum_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollTmpBn("fw_tmp");
}

const PbMessage& ReduceSumOp::GetCustomizedConf() const { return op_conf().reduce_sum_conf(); }

Maybe<void> ReduceSumOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
  CHECK_OR_RETURN(!conf.axis().empty());
  const BlobDesc* in_blob = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("fw_tmp") = *in_blob;
  BlobDesc* out_blob = GetBlobDesc4BnInOp("out");
  out_blob->set_data_type(in_blob->data_type());
  const AxisVector axis_vec = {conf.axis().begin(), conf.axis().end()};
  const Shape& reduced_shape = CreateReducedShape(in_blob->shape(), axis_vec);
  if (conf.keep_dims()) {
    out_blob->mut_shape() = reduced_shape;
  } else {
    out_blob->mut_shape() = reduced_shape.RemoveOnes(axis_vec);
  }
  return Maybe<void>::Ok();
}

Maybe<void> ReduceSumOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
  bool keep_dims = conf.keep_dims();
  HashSet<int64_t> conf_axes = {conf.axis().begin(), conf.axis().end()};
  const OptInt64* in_batch_axis = BatchAxis4BnInOp("in");
  OptInt64* out_batch_axis = BatchAxis4BnInOp("out");
  if (in_batch_axis->has_value()) {
    if (keep_dims || conf_axes.find(in_batch_axis->value()) == conf_axes.end()) {
      *out_batch_axis = *in_batch_axis;
    } else {
      out_batch_axis->clear_value();
    }
  } else {
    out_batch_axis->clear_value();
  }
  return Maybe<void>::Ok();
}

Maybe<void> ReduceSumOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int64_t num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
  auto IsReducedAxis =
      ReduceSbpUtil::MakePredicatorIsReducedAxis(op_conf().reduce_sum_conf().axis(), num_axes);
  int64_t num_reduced_axes = 0;
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (IsReducedAxis(i)) {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .PartialSum(output_bns())
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
      num_reduced_axes += 1;
    } else {
      SbpSignatureBuilder()
          .Split(input_bns(), i)
          .Split(output_bns(), op_conf().reduce_sum_conf().keep_dims() ? i : i - num_reduced_axes)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kReduceSumConf, ReduceSumOp);

}  // namespace oneflow
