#include "oneflow/core/operator/operator.h"

namespace oneflow {

class PieceSliceOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PieceSliceOp);
  PieceSliceOp() = default;
  ~PieceSliceOp() = default;

  void InitFromOpConf() {
    CHECK(op_conf().has_piece_slice_conf());
    EnrollInputBn("in");
    EnrollRepeatedOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().piece_slice_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override {
    const BlobDesc* in = GetBlobDesc4BnInOp(SoleIbn());

    // only used to run compile stage for test, remove later
    FOR_RANGE(int32_t, i, 0, op_conf().piece_slice_conf().out_size()) {
      BlobDesc* out_i = GetBlobDesc4BnInOp(output_bns().Get(i));
      out_i->mut_shape() = Shape(
          std::vector<int64_t>(in->shape().dim_vec().begin() + 1, in->shape().dim_vec().end()));
      out_i->set_data_type(in->data_type());
    }

    // correct logits
    // CHECK_EQ_OR_RETURN(in->num_of_lod_levels(), 2);
    // auto dim_vec = in->shape().dim_vec();
    // dim_vec.erase(dim_vec.begin());
    // FOR_RANGE(int32_t, i, 0, op_conf().piece_slice_conf().out_size()) {
    //   BlobDesc* out_i = GetBlobDesc4BnInOp(output_bns().Get(i));
    //   out_i->mut_shape() = Shape(dim_vec);
    //   out_i->set_data_type(in->data_type());
    //   out_i->set_is_dynamic(true);
    // }

    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
    for (const auto& obn : output_bns()) { *BatchAxis4BnInOp(obn) = *BatchAxis4BnInOp(SoleIbn()); }
    return Maybe<void>::Ok();
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const {
    // TODO
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kPieceSliceConf, PieceSliceOp);

}  // namespace oneflow
