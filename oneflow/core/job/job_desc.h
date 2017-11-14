#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  ~JobDesc() = default;

  OF_SINGLETON(JobDesc);

  void ToProto(JobDescProto*) const;

  // Common
  const JobConf& job_conf() const { return job_conf_; }
  const DLNetConf& dlnet_conf() const { return dlnet_conf_; }
  const Resource& resource() const { return resource_; }
  const Placement& placement() const { return placement_; }
  const std::string& md_load_snapshot_path() {
    return job_conf_.model_load_snapshot_path();
  }
  DataType default_data_type() const { return job_conf_.default_data_type(); }
  bool use_async_cpu_stream() const { return job_conf_.use_async_cpu_stream(); }
  size_t SizeOfOneDataId() const {
    return job_conf_.max_data_id_length() * sizeof(char);
  }
  int64_t TotalMachineNum() const { return resource_.machine().size(); }
  int32_t CommNetIOWorkerNum() const {
    return job_conf_.comm_net_io_worker_num();
  }
  bool is_train() const { return job_conf_.has_train_conf(); }
  bool is_predict() const { return job_conf_.has_predict_conf(); }

  // Train conf
  const std::string& md_save_snapshots_path() {
    CHECK(is_train());
    return job_conf_.train_conf().model_save_snapshots_path();
  }
  int32_t num_of_batches_in_snapshot() const {
    CHECK(is_train());
    return job_conf_.train_conf().num_of_batches_in_snapshot();
  }
  int32_t num_of_pieces_in_batch() const {
    CHECK(is_train());
    return job_conf_.train_conf().num_of_pieces_in_batch();
  }
  int32_t staleness() const {
    CHECK(is_train());
    return job_conf_.train_conf().staleness();
  }
  int64_t total_batch_num() const {
    CHECK(is_train());
    return job_conf_.train_conf().total_batch_num();
  }
  const FillConf* default_fill_conf() const {
    CHECK(is_train());
    return OF_PB_POINTER_GET(job_conf_.train_conf(), default_fill_conf);
  }
  int32_t piece_num_of_record_loss() const {
    CHECK(is_train());
    return job_conf_.train_conf().piece_num_of_record_loss();
  }
  int32_t SinglePieceSize() const { return job_conf_.single_piece_size(); }
  int32_t ParallelPieceSize() const {
    return job_conf_.data_part_num() * SinglePieceSize();
  }
  int32_t BatchSize() const {
    return num_of_pieces_in_batch() * ParallelPieceSize();
  }

 private:
  JobDesc(const JobConf&);
  JobDesc(const JobDescProto&);

  JobConf job_conf_;
  DLNetConf dlnet_conf_;
  Resource resource_;
  Placement placement_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
