#include "oneflow/core/operator/rnn_data_loader_op.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void RnnDataLoaderOp::InitFromOpConf() {
  CHECK(op_conf().has_rnn_data_loader_conf());
  
  EnrollOutputBn("out",false);
}

const PbMessage& RnnDataLoaderOp::GetSpecialConf() const {
  return op_conf().rnn_data_loader_conf();
}

void RnnDataLoaderOp::InferBlobDesc4FwBlobs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    ParallelPolicy policy, int64_t parallel_id, int64_t parallel_num) {
  const DataLoaderOpConf& conf = op_conf().rnn_data_loader_conf();
  BlobDesc* out = GetBlobDesc4BnInOp("out");
  out->mut_shape() = Shape(conf.shape());
  out->mut_shape().Set(0, JobDesc::Singleton()->piece_size());
  out->set_data_type(conf.data_type());
  out->set_has_data_id(JobDesc::Singleton()->is_predict()
                      && JobDesc::Singleton()->SizeofOneDataId() != 0);
}

REGISTER_OP(OperatorConf::kRnnDataLoaderConf, RnnDataLoaderOp);

} // namespace oneflow
