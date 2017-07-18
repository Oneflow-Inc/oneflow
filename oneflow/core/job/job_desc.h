#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/util.h"
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
  const Strategy& strategy() const { return strategy_; }
  const std::string& md_load_snapshot_path() {
    return job_conf_.model_load_snapshot_path();
  }
  const std::string& md_save_snapshots_path() {
    return job_conf_.model_save_snapshots_path();
  }
  int32_t piece_size() const { return job_conf_.piece_size(); }
  int32_t num_of_pieces_in_batch() const {
    return job_conf_.num_of_pieces_in_batch();
  }
  int32_t batch_size() const { return piece_size() * num_of_pieces_in_batch(); }
  bool is_train() const { return job_conf_.is_train(); }
  FloatingPointTypeProto floating_point_type() const {
    return job_conf_.floating_point_type();
  }
  size_t FloatingPointSize() const {
    if (floating_point_type() == FloatingPointTypeProto::kFloat) {
      return sizeof(float);
    } else if (floating_point_type() == FloatingPointTypeProto::kDouble) {
      return sizeof(double);
    } else {
      UNEXPECTED_RUN();
    }
  }
  int32_t num_of_batches_in_snapshot() const {
    return job_conf_.num_of_batches_in_snapshot();
  }
  int32_t staleness() const { return job_conf_.staleness(); }
  int64_t total_batch_num() const { return job_conf_.total_batch_num(); }
  int64_t total_piece_num() const {
    return total_batch_num() * num_of_pieces_in_batch();
  }

 private:
  JobDesc() = default;

  JobConf job_conf_;
  DLNetConf train_dlnet_conf_;
  Resource resource_;
  Strategy strategy_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
