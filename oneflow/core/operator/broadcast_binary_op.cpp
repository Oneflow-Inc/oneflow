#include "oneflow/core/operator/broadcast_binary_op.h"

namespace oneflow {

namespace {

bool IsScalarBlob(const BlobDesc* blob) {
  return blob->shape().NumAxes() == 1 && blob->shape().At(0) == 1;
}

std::unique_ptr<const OpParallelSignature> MakeBroadcastBinaryOpParallelSignature(
    const Operator* op, const HashSet<std::string>& model_input_bns) {
  std::string data_split_desc = op->op_name() + ": (C, ..., S(0), ...) -> (S(0), ...)";
  auto IsMatched =
      [op, model_input_bns](
          const std::function<const LogicalBlobParallelDesc&(const std::string&)>& ProducerLbpd4Ibn,
          const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
          const ParallelContext* parallel_ctx) {
        OpParallelMatchResult default_ret;
        if (parallel_ctx->policy() == kDataParallel) {
          default_ret = MakeOpParallelMatchSuccess();
        } else {
          default_ret =
              MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kDataParallel);
        }
        for (const auto& bn : op->input_bns()) {
          const auto& producer_lbpd = ProducerLbpd4Ibn(bn);
          bool is_model_input_bns = (model_input_bns.find(bn) != model_input_bns.end());
          bool has_actual_model_input =
              (producer_lbpd.has_clone_parallel() || ModelSplitAxis4BnInOp(bn) != -1);
          if (is_model_input_bns ^ has_actual_model_input) {
            return MakeOpParallelMatchSignatureMismatch();
          }
        }
        if (parallel_ctx->policy() == kDataParallel) { return MakeOpParallelMatchSuccess(); }
        return MakeOpParallelMatchParallelPolicyError(parallel_ctx->policy(), kDataParallel);
      };
  auto GenDataSplitSignature =
      [op, model_input_bns](const std::function<int32_t(const std::string&)>& ModelSplitAxis4BnInOp,
                            HashMap<std::string, LogicalBlobParallelDesc>* signature) {
        for (const auto& bn : op->input_bns()) {
          if (model_input_bns.find(bn) != model_input_bns.end()) {
            (*signature)[bn].mutable_clone_parallel();
          } else {
            (*signature)[bn].mutable_split_parallel()->set_axis(0);
          }
        }
        for (const auto& bn : op->output_bns()) {
          (*signature)[bn].mutable_split_parallel()->set_axis(0);
        }
      };
  return std::make_unique<OpParallelSignature>(data_split_desc, IsMatched, GenDataSplitSignature);
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
