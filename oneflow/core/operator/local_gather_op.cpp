#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

int64_t GetGatherAxis(const LocalGatherOpConf& conf, const int64_t num_axes) {
  const int64_t axis = conf.axis() < 0 ? num_axes + conf.axis() : conf.axis();
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return axis;
}

}  // namespace

class LocalGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalGatherOp);
  LocalGatherOp() = default;
  ~LocalGatherOp() override = default;

  void InitFromOpConf() override {
    CHECK(op_conf().has_local_gather_conf());
    EnrollInputBn("indices", false);
    EnrollInputBn("in");
    EnrollOutputBn("out");
  }

  const PbMessage& GetCustomizedConf() const override { return op_conf().local_gather_conf(); }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
                             std::function<void(OpContext*)> EnrollOpCtx) const override {
    const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
    CHECK_OR_RETURN(IsIntegralDataType(indices->data_type()));
    CHECK_GT_OR_RETURN(indices->shape().NumAxes(), 0);
    const BlobDesc* in = GetBlobDesc4BnInOp("in");
    CHECK_GT_OR_RETURN(in->shape().NumAxes(), 0);
    const int64_t axis = GetGatherAxis(op_conf().local_gather_conf(), in->shape().NumAxes());
    BlobDesc* out = GetBlobDesc4BnInOp("out");
    *out = *in;
    std::vector<int64_t> dim_vec;
    dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin(),
                   in->shape().dim_vec().cbegin() + axis);
    dim_vec.insert(dim_vec.end(), indices->shape().dim_vec().cbegin(),
                   indices->shape().dim_vec().cend());
    dim_vec.insert(dim_vec.end(), in->shape().dim_vec().cbegin() + axis + 1,
                   in->shape().dim_vec().end());
    out->mut_shape() = Shape(dim_vec);
    // if (axis == 0 && indices->has_dim0_valid_num_field()) {
    //   out->set_has_dim0_valid_num_field(true);
    //   out->mut_dim0_inner_shape() = Shape({1, indices->shape().At(0)});
    // } else if (axis > 0 && in->has_dim0_valid_num_field()) {
    //   out->set_has_dim0_valid_num_field(true);
    //   out->mut_dim0_inner_shape() = Shape({1, in->shape().At(0)});
    // } else {
    //   out->set_has_dim0_valid_num_field(false);
    // }
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    // const int64_t axis = GetGatherAxis(op_conf().local_gather_conf(), GetBlobDesc4BnInOp("in"));
    // kernel_conf->mutable_local_gather_conf()->set_axis(axis);
  }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override {
    const int64_t in_num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
    const int64_t gather_axis = GetGatherAxis(op_conf().local_gather_conf(), in_num_axes);
    CHECK_GE(gather_axis, 0);
    CHECK_LT(gather_axis, in_num_axes);
    const int64_t indices_num_axes = JUST(LogicalBlobDesc4Ibn("indices"))->shape().NumAxes();
    FOR_RANGE(int64_t, i, 0, indices_num_axes) {
      SbpSignatureBuilder()
          .Split("indices", i)
          .Broadcast("in")
          .Split("out", gather_axis + i)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
    FOR_RANGE(int64_t, i, 0, in_num_axes) {
      if (i == gather_axis) { continue; }
      SbpSignatureBuilder()
          .Broadcast("indices")
          .Split("in", i)
          .Split("out", i < gather_axis ? i : i + indices_num_axes - 1)
          .Build(sbp_sig_list->mutable_sbp_signature()->Add());
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_OP(OperatorConf::kLocalGatherConf, LocalGatherOp);

}  // namespace oneflow
