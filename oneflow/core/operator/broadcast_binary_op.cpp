#include "oneflow/core/operator/broadcast_binary_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

bool IsScalarBlob(const BlobDesc* blob) {
  return blob->shape().NumAxes() == 1 && blob->shape().At(0) == 1;
}

}  // namespace

void BroadcastBinaryOp::InitFromOpConf() {
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
}

Maybe<void> BroadcastBinaryOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  const BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");
  CHECK_EQ_OR_RETURN(a_blob_desc->data_type(), b_blob_desc->data_type());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  size_t output_num_axes = std::max(a_blob_desc->shape().NumAxes(), b_blob_desc->shape().NumAxes());
  if (IsScalarBlob(a_blob_desc)) {
    *out_blob_desc = *b_blob_desc;
  } else if (IsScalarBlob(b_blob_desc)) {
    *out_blob_desc = *a_blob_desc;
  } else {
    const auto& a_shape = a_blob_desc->shape().CreateLeftExtendedShape(output_num_axes);
    const auto& b_shape = b_blob_desc->shape().CreateLeftExtendedShape(output_num_axes);
    *out_blob_desc = *a_blob_desc;
    Shape out_shape(a_shape);
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes()) {
      CHECK_OR_RETURN(a_shape.At(i) == 1 || b_shape.At(i) == 1 || a_shape.At(i) == b_shape.At(i));
      out_shape.Set(i, std::max(a_shape.At(i), b_shape.At(i)));
    }
    out_blob_desc->mut_shape() = out_shape;
  }
  out_blob_desc->set_is_dynamic(a_blob_desc->is_dynamic() || b_blob_desc->is_dynamic());
  JUST(VirtualInferBlobDescs(GetBlobDesc4BnInOp));
  return Maybe<void>::Ok();
}

Maybe<void> BroadcastBinaryOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const Shape& a_shape = JUST(LogicalBlobDesc4Ibn("a"))->shape();
  const Shape& b_shape = JUST(LogicalBlobDesc4Ibn("b"))->shape();
  if (a_shape.NumAxes() < b_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, b_shape.NumAxes() - a_shape.NumAxes()) {
      SbpSignatureBuilder().Broadcast("a").Split("b", i).Split("out", i).Build(
          sbp_sig_list->mutable_sbp_signature()->Add());
    }
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes()) {
      SbpSignatureBuilder()
          .Split("a", a_shape.NumAxes() - 1 - i)
          .Split("b", b_shape.NumAxes() - 1 - i)
          .Split(output_bns(), b_shape.NumAxes() - 1 - i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  } else if (a_shape.NumAxes() > b_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes() - b_shape.NumAxes()) {
      SbpSignatureBuilder().Split("a", i).Broadcast("b").Split("out", i).Build(
          sbp_sig_list->mutable_sbp_signature()->Add());
    }
    FOR_RANGE(int64_t, i, 0, b_shape.NumAxes()) {
      SbpSignatureBuilder()
          .Split("a", a_shape.NumAxes() - 1 - i)
          .Split("b", b_shape.NumAxes() - 1 - i)
          .Split(output_bns(), a_shape.NumAxes() - 1 - i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
  } else {
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes()) {
      if (a_shape.At(i) == 1 && b_shape.At(i) == 1) { continue; }
      if (a_shape.At(i) == b_shape.At(i)) {
        SbpSignatureBuilder()
            .Split(input_bns(), i)
            .Split(output_bns(), i)
            .Build(sbp_sig_list->mutable_sbp_signature()->Add());
      } else if (a_shape.At(i) == 1) {
        SbpSignatureBuilder()
            .Broadcast("a")
            .Split("b", i)
            .Split(output_bns(), i)
            .Build(sbp_sig_list->mutable_sbp_signature()->Add());
      } else if (b_shape.At(i) == 1) {
        SbpSignatureBuilder()
            .Split("a", i)
            .Broadcast("b")
            .Split(output_bns(), i)
            .Build(sbp_sig_list->mutable_sbp_signature()->Add());
      } else {
        UNIMPLEMENTED();
      }
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
