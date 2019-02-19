#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

namespace {

bool IsScalarBlob(const BlobDesc* blob) {
  return blob->shape().NumAxes() == 1 && blob->shape().At(0) == 1;
}

class BroadcastBinaryOpParallelSignature final : public OpParallelSignature {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastBinaryOpParallelSignature);
  ~BroadcastBinaryOpParallelSignature() override = default;

  BroadcastBinaryOpParallelSignature(const Operator* op,
                                     const HashSet<std::string>& model_input_bns)
      : OpParallelSignature(op), model_input_bns_(model_input_bns) {}

  const std::string Description() const override {
    return op().op_name() + ": (C, ..., S(0), ...) -> (S(0), ...)";
  }

  const OpParallelMatchResult GetMatchResult(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelContext* parallel_ctx) const override {
    for (const auto& bn : op().input_bns()) {
      const auto& sbp_infer_hint = SbpInferHint4Ibn(bn);
      bool is_model_input_bns = (model_input_bns_.find(bn) != model_input_bns_.end());
      bool has_actual_model_input = sbp_infer_hint.is_model_blob();
      if (is_model_input_bns ^ has_actual_model_input) {
        return MakeOpParallelMatchSignatureMismatch();
      }
    }
    if (parallel_ctx->policy() == kDataParallel) { return MakeOpParallelMatchSuccess(); }
    return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kDataParallel);
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      HashMap<std::string, SbpParallel>* bn2sbp) const override {
    for (const auto& bn : op().input_bns()) {
      if (model_input_bns_.find(bn) != model_input_bns_.end()) {
        (*bn2sbp)[bn].mutable_broadcast_parallel();
      } else {
        const auto& in_sbp = SbpInferHint4Ibn(bn).sbp_parallel();
        if (in_sbp.has_broadcast_parallel()) {
          (*bn2sbp)[bn].mutable_broadcast_parallel();
        } else {
          (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
        }
      }
    }
    for (const auto& bn : op().output_bns()) {
      (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
    }
  }

 private:
  HashSet<std::string> model_input_bns_;
};

std::unique_ptr<const OpParallelSignature> MakeBroadcastBinaryOpParallelSignature(
    const Operator* op, const HashSet<std::string>& model_input_bns) {
  return std::unique_ptr<const OpParallelSignature>(
      new BroadcastBinaryOpParallelSignature(op, model_input_bns));
}

}  // namespace

void BroadcastBinaryOp::InitFromOpConf() {
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out");
  EnrollBwBufBn("bw_buf");
}

void BroadcastBinaryOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  const BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");
  CHECK_EQ(a_blob_desc->data_type(), b_blob_desc->data_type());
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
      CHECK(a_shape.At(i) == 1 || b_shape.At(i) == 1 || a_shape.At(i) == b_shape.At(i));
      out_shape.Set(i, std::max(a_shape.At(i), b_shape.At(i)));
    }
    out_blob_desc->mut_shape() = out_shape;
  }
}

void BroadcastBinaryOp::InferBwBufBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*) const {
  const BlobDesc* out = GetBlobDesc4BnInOp("out");
  BlobDesc* bw_buf = GetBlobDesc4BnInOp("bw_buf");
  bw_buf->mut_shape() = Shape({out->shape().elem_cnt()});
  bw_buf->set_data_type(out->data_type());
}

void BroadcastBinaryOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(MakeBroadcastBinaryOpParallelSignature(this, {}));
  op_parallel_signatures->emplace_back(MakeBroadcastBinaryOpParallelSignature(this, {"a"}));
  op_parallel_signatures->emplace_back(MakeBroadcastBinaryOpParallelSignature(this, {"b"}));
  op_parallel_signatures->emplace_back(MakeBroadcastBinaryOpParallelSignature(this, {"a", "b"}));
}

}  // namespace oneflow
