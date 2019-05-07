#include "oneflow/core/operator/local_gather_op.h"

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

void LocalGatherOp::InitFromOpConf() {
  CHECK(op_conf().has_local_gather_conf());
  EnrollInputBn("indices", false);
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& LocalGatherOp::GetCustomizedConf() const { return op_conf().local_gather_conf(); }

void LocalGatherOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  CHECK(IsIntegralDataType(indices->data_type()));
  CHECK_GT(indices->shape().NumAxes(), 0);
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  CHECK_GT(in->shape().NumAxes(), 0);
  const int64_t axis = GetGatherAxis(op_conf().local_gather_conf(), in);
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  *out = *in;
  auto dim_vec = in->shape().dim_vec();
  auto insert_dim_vec = indices->shape().dim_vec();
  auto insert_pos = dim_vec.erase(dim_vec.begin() + axis);
  dim_vec.insert(insert_pos, insert_dim_vec.begin(), insert_dim_vec.end());
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
}

void LocalGatherOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const int64_t axis = GetGatherAxis(op_conf().local_gather_conf(), GetBlobDesc4BnInOp("in"));
  kernel_conf->mutable_local_gather_conf()->set_axis(axis);
}

void LocalGatherOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kLocalGatherConf, LocalGatherOp);

}  // namespace oneflow
