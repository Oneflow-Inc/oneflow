#include "oneflow/core/operator/operator.h"

namespace oneflow {

namespace {

int64_t GetGatherAxis(const LocalGatherOpConf& conf, const BlobDesc* blob_desc) {
  const int64_t num_axes = blob_desc->shape().NumAxes();
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
    const int64_t axis = GetGatherAxis(op_conf().local_gather_conf(), in);
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
    if (axis == 0 && indices->has_dim0_valid_num_field()) {
      out->set_has_dim0_valid_num_field(true);
      out->mut_dim0_inner_shape() = Shape({1, indices->shape().At(0)});
    } else if (axis > 0 && in->has_dim0_valid_num_field()) {
      out->set_has_dim0_valid_num_field(true);
      out->mut_dim0_inner_shape() = Shape({1, in->shape().At(0)});
    } else {
      out->set_has_dim0_valid_num_field(false);
    }
    return Maybe<void>::Ok();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override {
    // const int64_t axis = GetGatherAxis(op_conf().local_gather_conf(), GetBlobDesc4BnInOp("in"));
    // kernel_conf->mutable_local_gather_conf()->set_axis(axis);
  }
};

REGISTER_OP(OperatorConf::kLocalGatherConf, LocalGatherOp);

}  // namespace oneflow
