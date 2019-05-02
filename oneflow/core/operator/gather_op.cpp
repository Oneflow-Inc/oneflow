#include "oneflow/core/operator/gather_op.h"
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

}  // namespace

Shape GatherGetOutShape(const Shape& in, const Shape& indices, const int64_t axis) {
  std::vector<int64_t> dim_vec;
  dim_vec.insert(dim_vec.end(), in.dim_vec().cbegin(), in.dim_vec().cbegin() + axis);
  dim_vec.insert(dim_vec.end(), indices.dim_vec().cbegin(), indices.dim_vec().cend());
  dim_vec.insert(dim_vec.end(), in.dim_vec().cbegin() + axis + 1, in.dim_vec().end());
  return Shape(dim_vec);
}

void GatherOp::InitFromOpConf() {
  CHECK(op_conf().has_gather_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& GatherOp::GetCustomizedConf() const { return op_conf().gather_conf(); }

void GatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GT(indices->shape().NumAxes(), 0);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), in);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  out->mut_shape() = Shape(GatherGetOutShape(in->shape(), indices->shape(), axis));
}

void GatherOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const int64_t axis = GetGatherAxis(op_conf().gather_conf(), GetBlobDesc4BnInOp("in"));
  kernel_conf->mutable_gather_conf()->set_axis(axis);
}

void GatherOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  CHECK_EQ(LogicalBlobDesc4Ibn("in").shape().NumAxes(), 2);
  int64_t gather_axis = GetGatherAxis(op_conf().gather_conf(), &LogicalBlobDesc4Ibn("in"));
  CHECK_EQ(gather_axis, 0);
  SbpSignatureBuilder()
      .Split("indices", 0)
      .Broadcast("in")
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  // TODO: complete other signatures
}

REGISTER_OP(OperatorConf::kGatherConf, GatherOp);

}  // namespace oneflow
