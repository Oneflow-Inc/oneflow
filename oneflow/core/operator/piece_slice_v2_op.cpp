#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PieceSliceV2Op final : public Operator {
 public:
  OF_DISALLOW_COPY(PieceSliceV2Op);
  PieceSliceV2Op() = default;
  ~PieceSliceV2Op() = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_piece_slice_v2_conf());
    EnrollInputBn("in");
    const int32_t out_size = op_conf().piece_slice_v2_conf().out_size();
    CHECK_GT(out_size, 0);
    EnrollRepeatedOutputBn("out", out_size);
  }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* in = GetBlobDesc4BnInOp(SoleIbn());
    CHECK_OR_RETURN(!in->has_dim0_valid_num_field());
    const int32_t out_size = op_conf().piece_slice_v2_conf().out_size();
    // out_size should be equal to in->shape().At(0), but record_piece_size is set to 1 in op_graph,
    // so we use global device piece size to do this CHECK_OR_RETURN
    // TODO: fix this
    // CHECK_EQ(out_size, Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(*parallel_ctx));

    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kPieceSliceV2Conf, PieceSliceV2Op);

}  // namespace oneflow
