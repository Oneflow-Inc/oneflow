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
  std::vector<int64_t> dim_vec(1 + conf.shape().dim_size());
  dim_vec[0] = JobDesc::Singleton()->SinglePieceSize();
  FOR_RANGE(size_t, i, 1, dim_vec.size()) {
    dim_vec[i] = conf.shape().dim(i - 1);
  }
  out->mut_shape() = Shape(dim_vec);
  out->set_data_type(conf.data_type());
  out->set_has_data_id(JobDesc::Singleton()->IsPredict()
                       && JobDesc::Singleton()->SizeOfOneDataId() != 0);
}

REGISTER_OP(OperatorConf::kDataLoaderConf, DataLoaderOp);

}  // namespace oneflow
