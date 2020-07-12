#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class UnsortedSegmentSumOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnsortedSegmentSumOp);
  UnsortedSegmentSumOp() = default;
  ~UnsortedSegmentSumOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

void UnsortedSegmentSumOp::InitFromOpConf() {
  CHECK(op_conf().has_unsorted_segment_sum_conf());
  EnrollInputBn("segment_ids", false);
  EnrollInputBn("data");
  EnrollOutputBn("out");
}

const PbMessage& UnsortedSegmentSumOp::GetCustomizedConf() const {
  return op_conf().unsorted_segment_sum_conf();
}

Maybe<void> UnsortedSegmentSumOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const UnsortedSegmentSumOpConf& conf = op_conf().unsorted_segment_sum_conf();
  const BlobDesc* segment_ids = GetBlobDesc4BnInOp("segment_ids");
  CHECK_OR_RETURN(IsIndexDataType(segment_ids->data_type()));
  const BlobDesc* data = GetBlobDesc4BnInOp("data");
  DimVector out_dim_vec;
  out_dim_vec.insert(out_dim_vec.end(), data->shape().dim_vec().cbegin(),
                     data->shape().dim_vec().cbegin() + conf.axis());
  out_dim_vec.push_back(conf.num_segments());
  out_dim_vec.insert(
      out_dim_vec.end(),
      data->shape().dim_vec().cbegin() + conf.axis() + segment_ids->shape().NumAxes(),
      data->shape().dim_vec().end());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->set_data_type(data->data_type());
  out->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> UnsortedSegmentSumOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t segment_sum_axis = op_conf().unsorted_segment_sum_conf().axis();
  const int64_t segment_ids_num_axes = JUST(LogicalBlobDesc4Ibn("segment_ids"))->shape().NumAxes();
  const int64_t data_num_axes = JUST(LogicalBlobDesc4Ibn("data"))->shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, segment_ids_num_axes) {
    SbpSignatureBuilder()
        .Split("segment_ids", i)
        .Split("data", i + segment_sum_axis)
        .PartialSum("out")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  FOR_RANGE(int64_t, i, 0, data_num_axes) {
    if (i >= segment_sum_axis && i < segment_sum_axis + segment_ids_num_axes) { continue; }
    const int64_t out_split_axis = (i < segment_sum_axis) ? i : i - segment_ids_num_axes + 1;
    if (out_split_axis == segment_sum_axis) { continue; }
    SbpSignatureBuilder()
        .Broadcast("segment_ids")
        .Split("data", i)
        .Split("out", out_split_axis)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .Broadcast("segment_ids")
      .PartialSum("data")
      .PartialSum("out")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

Maybe<void> UnsortedSegmentSumOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_OR_RETURN(*BatchAxis4BnInOp("data") == *BatchAxis4BnInOp("segment_ids"));
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

void UnsortedSegmentSumOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_unsorted_segment_sum_conf()->set_indices_data_type(
      GetBlobDesc4BnInOp("segment_ids")->data_type());
  kernel_conf->mutable_unsorted_segment_sum_conf()->set_axis(
      op_conf().unsorted_segment_sum_conf().axis());
}

class UnsortedSegmentSumLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnsortedSegmentSumLikeOp);
  UnsortedSegmentSumLikeOp() = default;
  ~UnsortedSegmentSumLikeOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

void UnsortedSegmentSumLikeOp::InitFromOpConf() {
  CHECK(op_conf().has_unsorted_segment_sum_like_conf());
  EnrollInputBn("segment_ids", false);
  EnrollInputBn("data");
  EnrollInputBn("like", false)->set_use_header_only(true);
  EnrollOutputBn("out");
}

const PbMessage& UnsortedSegmentSumLikeOp::GetCustomizedConf() const {
  return op_conf().unsorted_segment_sum_like_conf();
}

Maybe<void> UnsortedSegmentSumLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const UnsortedSegmentSumLikeOpConf& conf = op_conf().unsorted_segment_sum_like_conf();
  const BlobDesc* segment_ids = GetBlobDesc4BnInOp("segment_ids");
  CHECK_OR_RETURN(IsIndexDataType(segment_ids->data_type()));
  const BlobDesc* data = GetBlobDesc4BnInOp("data");
  const BlobDesc* like = GetBlobDesc4BnInOp("like");
  CHECK_EQ_OR_RETURN(data->data_type(), like->data_type());
  CHECK_GE_OR_RETURN(conf.axis(), 0);
  CHECK_LE_OR_RETURN(conf.axis(), like->shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, conf.axis()) {
    CHECK_EQ_OR_RETURN(like->shape().At(i), data->shape().At(i));
  }
  CHECK_EQ_OR_RETURN(data->shape().NumAxes() - segment_ids->shape().NumAxes() + 1,
                     like->shape().NumAxes());
  FOR_RANGE(int64_t, i, conf.axis() + 1, like->shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(like->shape().At(i),
                       data->shape().At(i + segment_ids->shape().NumAxes() - 1));
  }
  GetBlobDesc4BnInOp("out")->CopyMetaFrom(*like);
  return Maybe<void>::Ok();
}

Maybe<void> UnsortedSegmentSumLikeOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t segment_sum_axis = op_conf().unsorted_segment_sum_conf().axis();
  const int64_t segment_ids_num_axes = JUST(LogicalBlobDesc4Ibn("segment_ids"))->shape().NumAxes();
  const int64_t data_num_axes = JUST(LogicalBlobDesc4Ibn("data"))->shape().NumAxes();
  FOR_RANGE(int64_t, i, 0, segment_ids_num_axes) {
    SbpSignatureBuilder()
        .Split("segment_ids", i)
        .Split("data", i + segment_sum_axis)
        .Broadcast("like")
        .PartialSum("out")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    SbpSignatureBuilder()
        .Split("segment_ids", i)
        .Split("data", i + segment_sum_axis)
        .PartialSum("like")
        .PartialSum("out")
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  FOR_RANGE(int64_t, i, 0, data_num_axes) {
    if (i >= segment_sum_axis && i < segment_sum_axis + segment_ids_num_axes) { continue; }
    const int64_t out_split_axis = (i < segment_sum_axis) ? i : i - segment_ids_num_axes + 1;
    if (out_split_axis == segment_sum_axis) { continue; }
    SbpSignatureBuilder()
        .Broadcast("segment_ids")
        .Split("data", i)
        .Split("like", out_split_axis)
        .Split("out", out_split_axis)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .Broadcast("segment_ids")
      .PartialSum("data")
      .Broadcast("like")
      .PartialSum("out")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder()
      .Broadcast("segment_ids")
      .PartialSum("data")
      .PartialSum("like")
      .PartialSum("out")
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

Maybe<void> UnsortedSegmentSumLikeOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("like");
  return Maybe<void>::Ok();
}

void UnsortedSegmentSumLikeOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  kernel_conf->mutable_unsorted_segment_sum_conf()->set_indices_data_type(
      GetBlobDesc4BnInOp("segment_ids")->data_type());
  kernel_conf->mutable_unsorted_segment_sum_conf()->set_axis(
      op_conf().unsorted_segment_sum_like_conf().axis());
}

REGISTER_OP(OperatorConf::kUnsortedSegmentSumConf, UnsortedSegmentSumOp);
REGISTER_OP(OperatorConf::kUnsortedSegmentSumLikeConf, UnsortedSegmentSumLikeOp);

}  // namespace oneflow
