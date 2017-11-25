#include "oneflow/core/operator/data_loader_op.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void DataLoaderOp::InitFromOpConf() {
  CHECK(op_conf().has_data_loader_conf());

  EnrollOutputBn("out", false);
}

const PbMessage& DataLoaderOp::GetSpecialConf() const {
  return op_conf().data_loader_conf();
}

void DataLoaderOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const DataLoaderOpConf& conf = op_conf().data_loader_conf();
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape(conf.shape());
  out->mut_shape().Set(0, JobDesc::Singleton()->SinglePieceSize());
  out->set_data_type(conf.data_type());
  out->set_has_data_id(JobDesc::Singleton()->IsPredict()
                       && JobDesc::Singleton()->SizeOfOneDataId() != 0);
}

void DataLoaderOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  int64_t parallel_id = parallel_ctx->parallel_id();
  kernel_conf->mutable_data_loader_conf()->set_parallel_id(parallel_id);
}

REGISTER_OP(OperatorConf::kDataLoaderConf, DataLoaderOp);

}  // namespace oneflow
