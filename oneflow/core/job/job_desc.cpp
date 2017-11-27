#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"

namespace oneflow {

JobDesc::JobDesc(const JobConf& conf) {
  LOG(INFO) << "Init JobDesc from JobConf";
  job_conf_ = conf;
  ParseProtoFromTextFile(conf.dlnet_filepath(), &dlnet_conf_);
  ParseProtoFromTextFile(conf.resource_filepath(), &resource_);
  ParseProtoFromTextFile(conf.placement_filepath(), &placement_);
}

JobDesc::JobDesc(const JobDescProto& proto) {
  LOG(INFO) << "Init JobDesc from Proto";
  job_conf_ = proto.job_conf();
  dlnet_conf_ = proto.dlnet_conf();
  resource_ = proto.resource();
  placement_ = proto.placement();
}

void JobDesc::ToProto(JobDescProto* proto) const {
  *(proto->mutable_job_conf()) = job_conf_;
  *(proto->mutable_dlnet_conf()) = dlnet_conf_;
  *(proto->mutable_resource()) = resource_;
  *(proto->mutable_placement()) = placement_;
}

const std::string& JobDesc::MdLoadSnapshotPath() {
  return job_conf_.model_load_snapshot_path();
}
size_t JobDesc::SizeOfOneDataId() const {
  return job_conf_.max_data_id_length() * sizeof(char);
}
int32_t JobDesc::PersistenceWorkerNum() const {
  return resource_.persistence_worker_num();
}
int32_t JobDesc::BoxingWorkerNum() const {
  return resource_.boxing_worker_num();
}
int32_t JobDesc::CommNetWorkerNum() const {
  return resource_.comm_net_worker_num();
}

const std::string& JobDesc::MdSaveSnapshotsPath() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().model_save_snapshots_path();
}
int32_t JobDesc::NumOfBatchesInSnapshot() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().num_of_batches_in_snapshot();
}
int32_t JobDesc::NumOfPiecesInBatch() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().num_of_pieces_in_batch();
}
int32_t JobDesc::Staleness() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().staleness();
}
int64_t JobDesc::TotalBatchNum() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().total_batch_num();
}
const FillConf* JobDesc::DefaultFillConf() const {
  CHECK(IsTrain());
  return OF_PB_POINTER_GET(job_conf_.train_conf(), default_fill_conf);
}
int32_t JobDesc::PieceNumOfPrintLoss() const {
  CHECK(IsTrain());
  return job_conf_.train_conf().piece_num_of_print_loss();
}
int32_t JobDesc::ParallelPieceSize() const {
  return job_conf_.data_part_num() * SinglePieceSize();
}
int32_t JobDesc::BatchSize() const {
  return NumOfPiecesInBatch() * ParallelPieceSize();
}

}  // namespace oneflow
