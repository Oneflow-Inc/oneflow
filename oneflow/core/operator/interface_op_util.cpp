#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

void CheckShape(const Shape& shape) {
  FOR_RANGE(int, i, 1, shape.NumAxes()) { CHECK_GT(shape.At(i), 0); }
}

bool GetSplitAxis(const InterfaceBlobConf& input_blob_conf, size_t* split_axis) {
  if (input_blob_conf.has_split_axis()) {
    int64_t axis = input_blob_conf.split_axis();
    if (axis < 0) { axis += input_blob_conf.shape().dim_size(); }
    if (axis >= 0 && axis < input_blob_conf.shape().dim_size()) {
      *split_axis = axis;
      return true;
    }
  } else if (input_blob_conf.broadcast() == false && input_blob_conf.batch_axis().has_value()) {
    *split_axis = input_blob_conf.batch_axis().value();
    return true;
  }
  return false;
}

Maybe<void> GetSbpSignature(const InterfaceBlobConf& blob_conf, const PbRpf<std::string>& input_bns,
                            const PbRpf<std::string>& output_bns, SbpSignature* sbp_signature,
                            bool is_for_input_op) {
  auto BuildSbpSignature = [&](int64_t split_axis) {
    SbpSignatureBuilder sbp_signature_builder;
    if (is_for_input_op) {
      sbp_signature_builder.Broadcast(input_bns);
    } else {
      sbp_signature_builder.Split(input_bns, split_axis);
    }
    sbp_signature_builder.Split(output_bns, split_axis).Build(sbp_signature);
  };
  if (blob_conf.has_split_axis()) {
    int64_t num_axes = blob_conf.shape().dim_size();
    int64_t split_axis = blob_conf.split_axis();
    if (split_axis < 0) { split_axis += num_axes; }
    CHECK_GE_OR_RETURN(split_axis, 0);
    CHECK_LT_OR_RETURN(split_axis, num_axes);
    BuildSbpSignature(split_axis);
  } else if (blob_conf.has_broadcast()) {
    SbpSignatureBuilder().Broadcast(input_bns).Broadcast(output_bns).Build(sbp_signature);
  } else if (blob_conf.batch_axis().has_value()) {
    BuildSbpSignature(blob_conf.batch_axis().value());
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> InterfaceOpUtil::InferOutBlobDesc(const InterfaceBlobConf& blob_conf,
                                              BlobDesc* out_blob_desc,
                                              const ParallelContext* parallel_ctx) {
  return InferOutBlobDesc(blob_conf, out_blob_desc, parallel_ctx, 0);
}
Maybe<void> InterfaceOpUtil::InferOutBlobDesc(const InterfaceBlobConf& blob_conf,
                                              BlobDesc* out_blob_desc,
                                              const ParallelContext* parallel_ctx,
                                              int64_t record_piece_size) {
  out_blob_desc->mut_shape() = Shape(blob_conf.shape());
  CheckShape(out_blob_desc->shape());
  CHECK_GT(out_blob_desc->mut_shape().At(0), 0);
  if (blob_conf.has_data_type()) {
    out_blob_desc->set_data_type(blob_conf.data_type());
  } else {
    out_blob_desc->set_data_type(GlobalJobDesc().DefaultDataType());
  }
  out_blob_desc->set_has_dim0_valid_num_field(blob_conf.has_dim0_valid_num());
  if (blob_conf.has_dim0_inner_shape()) {
    CHECK(blob_conf.has_dim0_valid_num());
    out_blob_desc->mut_dim0_inner_shape() = Shape(blob_conf.dim0_inner_shape());
  }
  size_t split_axis = 0;
  if (GetSplitAxis(blob_conf, &split_axis)) {
    BalancedSplitter bs(out_blob_desc->shape().At(split_axis), parallel_ctx->parallel_num());
    out_blob_desc->mut_shape().Set(split_axis, bs.At(parallel_ctx->parallel_id()).size());
  }
  return Maybe<void>::Ok();
}

Maybe<void> InterfaceOpUtil::InferBatchAxis(const InterfaceBlobConf& blob_conf,
                                            OptInt64* batch_axis) {
  *batch_axis = blob_conf.batch_axis();
  return Maybe<void>::Ok();
}

Maybe<void> InterfaceOpUtil::GetInputLikeOpSbpSignature(const InterfaceBlobConf& blob_conf,
                                                        const PbRpf<std::string>& input_bns,
                                                        const PbRpf<std::string>& output_bns,
                                                        SbpSignature* sbp_signature) {
  GetSbpSignature(blob_conf, input_bns, output_bns, sbp_signature, true);
  return Maybe<void>::Ok();
}

Maybe<void> InterfaceOpUtil::GetOutputLikeOpSbpSignature(const InterfaceBlobConf& blob_conf,
                                                         const PbRpf<std::string>& input_bns,
                                                         const PbRpf<std::string>& output_bns,
                                                         SbpSignature* sbp_signature) {
  GetSbpSignature(blob_conf, input_bns, output_bns, sbp_signature, false);
  return Maybe<void>::Ok();
}

Maybe<void> InterfaceOpUtil::InitBlobConf(InterfaceBlobConf* blob_conf,
                                          const ParallelBlobConf& parallel_blob_conf) {
  BlobDesc blob_desc(parallel_blob_conf.logical_blob_desc_conf());
  blob_desc.shape().ToProto(blob_conf->mutable_shape());
  blob_conf->set_data_type(blob_desc.data_type());
  if (blob_desc.has_dim0_inner_shape()) {
    blob_desc.dim0_inner_shape().ToProto(blob_conf->mutable_dim0_inner_shape());
  }
  blob_conf->set_has_dim0_valid_num(blob_desc.has_dim0_valid_num_field());
  blob_conf->set_has_dim1_valid_num(blob_desc.has_dim1_valid_num_field());
  blob_conf->set_has_dim2_valid_num(blob_desc.has_dim2_valid_num_field());
  if (parallel_blob_conf.sbp_conf().has_split_parallel()) {
    blob_conf->set_split_axis(parallel_blob_conf.sbp_conf().split_parallel().axis());
  } else if (parallel_blob_conf.sbp_conf().has_broadcast_parallel()) {
    blob_conf->set_broadcast(true);
  } else {
    UNIMPLEMENTED();
  }
  *blob_conf->mutable_batch_axis() = parallel_blob_conf.batch_axis();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
