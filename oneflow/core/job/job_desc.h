#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"

namespace oneflow {

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  ~JobDesc() = default;

  OF_SINGLETON(JobDesc);

  void InitFromJobConf(const JobConf&);
  void InitFromProto(const JobDescProto&);
  void ToProto(JobDescProto*) const;

  // Getters
  const JobConf& job_conf() const { return job_conf_; }
  const DLNetConf& train_dlnet_conf() const { return train_dlnet_conf_; }
  const Resource& resource() const { return resource_; }
  const Placement& placement() const { return placement_; }
  const std::string& md_load_snapshot_path() {
    return job_conf_.model_load_snapshot_path();
  }
  const std::string& md_save_snapshots_path() {
    return job_conf_.train_conf().model_save_snapshots_path();
  }
  int32_t piece_size() const { return job_conf_.piece_size(); }
  int32_t num_of_pieces_in_batch() const {
    return job_conf_.num_of_pieces_in_batch();
  }
  int32_t batch_size() const { return piece_size() * num_of_pieces_in_batch(); }
  bool is_train() const { return job_conf_.has_train_conf(); }
  DataType default_data_type() const { return job_conf_.default_data_type(); }
  int32_t num_of_batches_in_snapshot() const {
    return job_conf_.train_conf().num_of_batches_in_snapshot();
  }
  int32_t staleness() const { return job_conf_.train_conf().staleness(); }
  int64_t total_batch_num() const {
    return job_conf_.train_conf().total_batch_num();
  }
  int64_t total_piece_num() const {
    return total_batch_num() * num_of_pieces_in_batch();
  }
  const FillConf* default_fill_conf() const {
    return OF_PB_POINTER_GET(job_conf_.train_conf(), default_fill_conf);
  }
  bool use_async_cpu_stream() const { return job_conf_.use_async_cpu_stream(); }
  int32_t piece_num_of_record_loss() const {
    return job_conf_.train_conf().piece_num_of_record_loss();
  }
  size_t SizeOfOneDataId() const {
    return job_conf_.max_data_id_length() * sizeof(char);
  }

 private:
  JobDesc() = default;

  JobConf job_conf_;
  DLNetConf train_dlnet_conf_;
  Resource resource_;
  Placement placement_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
