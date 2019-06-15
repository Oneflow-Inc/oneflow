#include "oneflow/core/operator/local_scatter_nd_update_op.h"

namespace oneflow {

void LocalScatterNdUpdateOp::InitFromOpConf() {
  CHECK(op_conf().has_local_scatter_nd_update_conf());
  EnrollInputBn("in");
  EnrollInputBn("indices", false);
  EnrollInputBn("updates");
  if (this->device_type(), DeviceType::kGPU) {
    // For better performance, input/output shape information is allocated on global memory, instead
    // of passing as a argument to cuda kernel.
    EnrollDataTmpBn("shape");
  }
  EnrollOutputBn("out");
}

const PbMessage& LocalScatterNdUpdateOp::GetCustomizedConf() const {
  return op_conf().local_scatter_nd_update_conf();
}

void LocalScatterNdUpdateOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  LocalScatterNdUpdateConf* conf = kernel_conf->mutable_local_scatter_nd_update_conf();
  conf->set_value_type(GetBlobDesc4BnInOp("in")->data_type());
  conf->set_indices_type(GetBlobDesc4BnInOp("indices")->data_type());
}

void LocalScatterNdUpdateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // input: in, indices, updates
  const BlobDesc* in = GetBlobDesc4BnInOp("in");
  const BlobDesc* indices = GetBlobDesc4BnInOp("indices");
  const BlobDesc* updates = GetBlobDesc4BnInOp("updates");
  CHECK_EQ(in->data_type(), updates->data_type());
  CHECK(IsIntegralDataType(indices->data_type()));
  FOR_RANGE(size_t, i, 0, indices->shape().NumAxes() - 1) {
    CHECK_EQ(indices->shape().At(i), updates->shape().At(i));
  }
  const auto indices_dim_vec = indices->shape().dim_vec();
  CHECK_LE(indices_dim_vec.back(), in->shape().NumAxes());
  FOR_RANGE(size_t, i, indices_dim_vec.back(), in->shape().NumAxes()) {
    CHECK_EQ(in->shape().At(i), updates->shape().At(i));
  }
  if (this->device_type(), DeviceType::kGPU) {
    // datatmp
    BlobDesc* shape = GetBlobDesc4BnInOp("shape");
    shape->mut_shape() = Shape({in->shape().NumAxes()});
    shape->set_data_type(DataType::kInt64);
  }

  // output
  *GetBlobDesc4BnInOp("out") = *in;
}

void LocalScatterNdUpdateOp::GetOpParallelSignatures(
    std::vector<std::unique_ptr<const OpParallelSignature>>* op_parallel_signatures) const {
  op_parallel_signatures->emplace_back(MakeDataSplitOpParallelSignature(this));
}

REGISTER_OP(OperatorConf::kLocalScatterNdUpdateConf, LocalScatterNdUpdateOp);

}  // namespace oneflow
