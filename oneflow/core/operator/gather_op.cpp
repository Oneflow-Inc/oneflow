#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

int64_t GetGatherAxis(const GatherOpConf& conf, int64_t num_axes) {
  const int64_t axis = conf.axis() < 0 ? num_axes + conf.axis() : conf.axis();
  CHECK_GE(axis, 0);
  CHECK_LT(axis, num_axes);
  return axis;
}

int64_t GetGatherAxis(const GatherOpConf& conf, const BlobDesc* in_blob_desc) {
  return GetGatherAxis(conf, in_blob_desc->shape().NumAxes());
}

Shape GatherGetOutShape(const Shape& in, const Shape& indices, const int64_t axis) {
  DimVector dim_vec;
  dim_vec.insert(dim_vec.end(), in.dim_vec().cbegin(), in.dim_vec().cbegin() + axis);
  dim_vec.insert(dim_vec.end(), indices.dim_vec().cbegin(), indices.dim_vec().cend());
  dim_vec.insert(dim_vec.end(), in.dim_vec().cbegin() + axis + 1, in.dim_vec().end());
  return Shape(dim_vec);
}

}  // namespace

class GatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherOp);
  GatherOp() = default;
  ~GatherOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }
  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetCustomizedConf() const { return op_conf().gather_conf(); }

Maybe<void> GatherOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIndexDataType(indices->data_type()));
  CHECK_GT(indices->shape().NumAxes(), 0);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), in);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->set_is_dynamic(indices->is_dynamic());
  out->mut_shape() = Shape(GatherGetOutShape(in->shape(), indices->shape(), axis));
  return Maybe<void>::Ok();
}

void GatherOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), GetBlobDesc4BnInOp("in"));
  kernel_conf->mutable_gather_conf()->set_axis(axis);
}

Maybe<void> GatherOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  const int64_t in_num_axes = JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes();
  const int64_t gather_axis = GetGatherAxis(op_conf().gather_conf(), in_num_axes);
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

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
