#include "oneflow/core/operator/slice_grad_op.h"

namespace oneflow {

void SliceGradOp::InitFromOpConf() {
  CHECK(op_conf().has_slice_grad_conf());
  EnrollInputBn("dy");
  EnrollInputBn("like")->set_use_header_only(true);
  EnrollOutputBn("dx");
  if (op_conf().device_type() == DeviceType::kGPU) { EnrollConstBufBn("y_to_x_offset"); }
}

const PbMessage& SliceGradOp::GetCustomizedConf() const { return op_conf().slice_grad_conf(); }

void SliceGradOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  const Shape& in_shape = GetBlobDesc4BnInOp("like")->shape();
  in_shape.ToProto(kernel_conf->mutable_slice_conf()->mutable_in_shape());
}

void SliceGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                 const ParallelContext* parallel_ctx) const {
  const SliceGradOpConf& conf = op_conf().slice_grad_conf();
  const BlobDesc* like_blob_desc = GetBlobDesc4BnInOp("like");
  CHECK_EQ(conf.dim_slice_conf_size(), like_blob_desc->shape().NumAxes() - 1);
  *GetBlobDesc4BnInOp("dx") = *like_blob_desc;
  if (op_conf().device_type() == DeviceType::kGPU) {
    BlobDesc* offset_blob_desc = GetBlobDesc4BnInOp("y_to_x_offset");
    *offset_blob_desc = *GetBlobDesc4BnInOp("dy");
    offset_blob_desc->set_data_type(DataType::kInt64);
  }
}

REGISTER_OP(OperatorConf::kSliceGradConf, SliceGradOp);

}  // namespace oneflow
