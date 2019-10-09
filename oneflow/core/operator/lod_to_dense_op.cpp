#include "oneflow/core/operator/operator.h"

namespace oneflow {

class LoDToDenseOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LoDToDenseOp);
  LoDToDenseOp() = default;
  ~LoDToDenseOp() override = default;

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
};

void LoDToDenseOp::InitFromOpConf() {
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_const_inplace_ibn("in");
}

Maybe<void> LoDToDenseOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc& in_blob_desc = *GetBlobDesc4BnInOp("in");
  int64_t num_of_lod_levels = in_blob_desc.num_of_lod_levels();
  OF_CHECK_GT(num_of_lod_levels, 0);
  OF_CHECK(in_blob_desc.is_dynamic());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = in_blob_desc;
  out_blob_desc->ClearLoD();
  {
    const auto& in_dim_vec = in_blob_desc.shape().dim_vec();
    std::vector<int64_t> dim_vec{in_blob_desc.shape().Count(0, num_of_lod_levels)};
    dim_vec.insert(dim_vec.end(), in_dim_vec.begin() + num_of_lod_levels, in_dim_vec.end());
    out_blob_desc->mut_shape() = Shape(dim_vec);
  }
  return Maybe<void>::Ok();
}

const PbMessage& LoDToDenseOp::GetCustomizedConf() const { return op_conf().lod_to_dense_conf(); }

Maybe<void> LoDToDenseOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return NaiveInferBatchAxis(BatchAxis4BnInOp);
}

Maybe<void> LoDToDenseOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const auto bns = StdVec2PbRpf<std::string>({"in", "out"});
  SbpSignatureBuilder().PartialSum(bns).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  const auto& in_blob_desc = *JUST(LogicalBlobDesc4Ibn("out"));
  const int64_t num_axes = JUST(LogicalBlobDesc4Ibn("out"))->shape().NumAxes();
  FOR_RANGE(int64_t, out_axis, 0, num_axes) {
    int64_t in_axis = (out_axis > 0 ? (out_axis + in_blob_desc.num_of_lod_levels()) : out_axis);
    SbpSignatureBuilder()
        .Split("in", in_axis)
        .Split("out", out_axis)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kLodToDenseConf, LoDToDenseOp);

}  // namespace oneflow
