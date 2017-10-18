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

void DataLoaderOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  const DataLoaderOpConf& conf = op_conf().data_loader_conf();
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape(conf.shape());
  out->mut_shape().Set(0, JobDesc::Singleton()->piece_size());
  out->set_data_type(conf.data_type());
  out->set_has_data_id(JobDesc::Singleton()->is_predict()
                       && JobDesc::Singleton()->SizeOfOneDataId() != 0);
}

REGISTER_OP(OperatorConf::kDataLoaderConf, DataLoaderOp);

}  // namespace oneflow
