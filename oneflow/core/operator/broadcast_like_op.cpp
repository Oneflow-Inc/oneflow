#include "oneflow/core/operator/broadcast_like_op.h"

namespace oneflow {
namespace {

bool IsScalarBlob(const BlobDesc* blob) {
  return blob->shape().NumAxes() == 1 && blob->shape().At(0) == 1;
}

class BroadcastLikeOpParallelSignature final : public OpParallelSignature {
 public:
  ~BroadcastLikeOpParallelSignature() override = default;

  const std::string Description() const override { return std::string(); }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    return OpParallelMatchResult();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {}
};

}  // namespace

void BroadcastLikeOp::InitFromOpConf() {
  EnrollInputBn("x");
  EnrollInputBn("like")->set_use_header_only(true);
  EnrollOutputBn("y");
}

const PbMessage& BroadcastLikeOp::GetCustomizedConf() const {
  return op_conf().broadcast_like_conf();
}

void BroadcastLikeOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("x");
  const BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("like");
  CHECK_EQ(a_blob_desc->data_type(), b_blob_desc->data_type());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("y");
  size_t output_num_axes = std::max(a_blob_desc->shape().NumAxes(), b_blob_desc->shape().NumAxes());
  if (IsScalarBlob(a_blob_desc)) {
    *out_blob_desc = *b_blob_desc;
  } else if (IsScalarBlob(b_blob_desc)) {
    *out_blob_desc = *a_blob_desc;
  } else {
    const auto& a_shape = op_conf().broadcast_like_conf().has_kept_dims_shape()
                              ? Shape(op_conf().broadcast_like_conf().kept_dims_shape())
                              : a_blob_desc->shape().CreateLeftExtendedShape(output_num_axes);
    const auto& b_shape = b_blob_desc->shape().CreateLeftExtendedShape(output_num_axes);
    *out_blob_desc = *a_blob_desc;
    Shape out_shape(a_shape);
    FOR_RANGE(int64_t, i, 0, a_shape.NumAxes()) {
      CHECK(a_shape.At(i) == 1 || b_shape.At(i) == 1 || a_shape.At(i) == b_shape.At(i));
      out_shape.Set(i, std::max(a_shape.At(i), b_shape.At(i)));
    }
    out_blob_desc->mut_shape() = out_shape;
  }
}

void BroadcastLikeOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {}

REGISTER_OP(OperatorConf::kBroadcastLikeConf, BroadcastLikeOp);

}  // namespace oneflow
