#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

class BatchGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchGatherOp);
  BatchGatherOp() = default;
  ~BatchGatherOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
};

void BatchGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_batch_gather_conf());
  EnrollInputBn("in");
  EnrollInputBn("indices", false);
  EnrollOutputBn("out");
}

const PbMessage& BatchGatherOp::GetCustomizedConf() const { return op_conf().batch_gather_conf(); }

Maybe<void> BatchGatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK_GT_OR_RETURN(indices->shape().NumAxes(), 0);
  CHECK_OR_RETURN(IsIndexDataType(indices->data_type()));
  const std::vector<int64_t>& in_dim_vec = in->shape().dim_vec();
  const std::vector<int64_t>& indices_dim_vec = indices->shape().dim_vec();
  CHECK_LE_OR_RETURN(indices_dim_vec.size(), in_dim_vec.size());
  FOR_RANGE(int64_t, i, 0, indices_dim_vec.size() - 1) {
    CHECK_EQ_OR_RETURN(indices_dim_vec.at(i), in_dim_vec.at(i));
  }
  // out
  std::vector<int64_t> out_dim_vec(indices_dim_vec);
  out_dim_vec.insert(out_dim_vec.end(), in_dim_vec.cbegin() + indices_dim_vec.size(),
                     in_dim_vec.cend());
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape() = Shape(out_dim_vec);
  return Maybe<void>::Ok();
}

Maybe<void> BatchGatherOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t indices_num_axes = JUST(LogicalBlobDesc4Ibn("indices"))->shape().NumAxes();
  if (indices_num_axes > 1) {
    FOR_RANGE(int64_t, i, 0, indices_num_axes - 1) {
      SbpSignatureBuilder()
          .Split("indices", i)
          .Split("in", i)
          .Split("out", i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
    SbpSignatureBuilder().Broadcast("indices").PartialSum("in").PartialSum("out").Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  } else {
    std::shared_ptr<ErrorProto> err;
    err->set_msg("BatchGatherOp: indices_num_axes equals " + std::to_string(indices_num_axes)
                 + " (should be bigger than 1).");
    err->mutable_check_failed();
    return err;
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBatchGatherConf, BatchGatherOp);

}  // namespace oneflow
