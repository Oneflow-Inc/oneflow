#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

namespace {

bool IsScalarBlob(const BlobDesc* blob) {
  return blob->shape().NumAxes() == 1 && blob->shape().At(0) == 1;
}

class BroadcastBinarySbpSignatureRule final : public ParallelSbpSignatureRule {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastBinarySbpSignatureRule);
  ~BroadcastBinarySbpSignatureRule() override = default;

  BroadcastBinarySbpSignatureRule(const Operator* op, const HashSet<std::string>& broadcast_bns)
      : ParallelSbpSignatureRule(op), broadcast_bns_(broadcast_bns) {}

  const std::string Description() const override {
    return op().op_name() + ": (C, ..., S(0), ...) -> (S(0), ...)";
  }

  const SbpSigMatchResult MatchByIbnHint(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override {
    return MakeSbpSigMatchSuccess();
  }

  void GenerateSignature(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      SbpSignature* sbp_signature) const override {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    auto SetSbpParallel = [&](const std::string& bn) {
      if (broadcast_bns_.find(bn) != broadcast_bns_.end()) {
        (*bn2sbp)[bn].mutable_broadcast_parallel();
      } else {
        (*bn2sbp)[bn].mutable_split_parallel()->set_axis(0);
      }
    };
    for (const auto& bn : op().input_bns()) { SetSbpParallel(bn); }
    for (const auto& bn : op().output_bns()) { SetSbpParallel(bn); }
  }

 private:
  HashSet<std::string> broadcast_bns_;
};

std::unique_ptr<const SbpSignatureRule> MakeBroadcastBinarySbpSignatureRule(
    const Operator* op, const HashSet<std::string>& broadcast_bns) {
  return std::unique_ptr<const SbpSignatureRule>(
      new BroadcastBinarySbpSignatureRule(op, broadcast_bns));
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

void BroadcastBinaryOp::GetSbpSignatureRules(
    const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
    std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const {
  HashSet<std::string> broadcast_bns;
  const auto& a_shape = SbpInferHint4Ibn("a").logical_blob_desc().shape();
  const auto& b_shape = SbpInferHint4Ibn("b").logical_blob_desc().shape();
  if (a_shape.NumAxes() != b_shape.NumAxes()) {
    broadcast_bns.insert(a_shape.NumAxes() < b_shape.NumAxes() ? "a" : "b");
  } else if (a_shape.At(0) == 1 && b_shape.At(0) == 1) {
    broadcast_bns = HashSet<std::string>{"a", "b", "out"};
  } else if (a_shape.At(0) == b_shape.At(0)) {
    // do nothing
  } else {
    broadcast_bns.insert(a_shape.At(0) < b_shape.At(0) ? "a" : "b");
  }
  rules->emplace_back(MakeBroadcastBinarySbpSignatureRule(this, broadcast_bns));
}

}  // namespace oneflow
