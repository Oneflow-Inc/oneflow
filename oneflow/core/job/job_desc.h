#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {

bool IsInterfaceOpConf(const OperatorConf& op_conf);

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  JobDesc(const JobConfigProto& job_conf, int64_t job_id);
  ~JobDesc() = default;
  // Common
  int64_t job_id() const { return job_id_; }
  const std::string& job_name() const { return job_conf_.job_name(); }
  int64_t concurrency_width() const { return job_conf_.concurrency_width(); }
  const JobConfigProto& job_conf() const { return job_conf_; }
  DataType DefaultDataType() const { return job_conf_.default_data_type(); }
  size_t SizeOfOneDataId() const { return job_conf_.max_data_id_length() * sizeof(char); }
  bool EnableCudnn() const { return job_conf_.enable_cudnn(); }
  bool IsTrain() const { return job_conf_.has_train_conf(); }
  bool IsPredict() const { return job_conf_.has_predict_conf(); }
  int64_t RecordPieceSize() const { return job_conf_.piece_size(); }
  int64_t piece_num_of_experiment_phase() const;
  bool enable_experiment_run() const;
  float available_zone_mem_ratio() const;
  size_t persistence_buf_byte() const;
  size_t reserved_host_mem_byte() const;
  size_t reserved_device_mem_byte() const;
  bool save_downloaded_file_to_local_fs() const;
  size_t rdma_mem_block_byte() const;
  size_t rdma_recv_msg_buf_byte() const;
  bool enable_mem_sharing() const { return job_conf_.enable_mem_sharing(); }
  bool enable_inplace() const { return job_conf_.enable_inplace(); }
  bool enable_true_half_config_when_conv() const {
    return job_conf_.enable_true_half_config_when_conv();
  }
  bool enable_float_compute_for_half_gemm() const {
    return job_conf_.enable_float_compute_for_half_gemm();
  }
  bool enable_auto_mixed_precision() const { return job_conf_.enable_auto_mixed_precision(); }
  const FileSystemConf& data_fs_conf() const;
  const FileSystemConf& snapshot_fs_conf() const;
  bool enable_write_snapshot() const;
  bool write_snapshot_to_master() const { return snapshot_fs_conf().has_localfs_conf(); }
  bool enable_nccl() const { return job_conf_.enable_nccl(); }
  bool use_nccl_inter_node_communication() const {
    return job_conf_.use_nccl_inter_node_communication();
  }
  bool enable_all_reduce_group() const { return job_conf_.enable_all_reduce_group(); }
  int64_t all_reduce_group_num() const;
  int64_t all_reduce_group_min_byte() const;
  float all_reduce_group_size_warmup() const;
  float all_reduce_lazy_ratio() const;
  bool all_reduce_fp16() const;
  int64_t cudnn_buf_limit_mbyte() const { return job_conf_.cudnn_buf_limit_mbyte(); }

  // Train conf
  int32_t NumOfBatchesInSnapshot() const;
  int64_t TotalBatchNum() const;
  int64_t BatchSize() const;
  int64_t NumOfPiecesInBatch() const;
  float primary_lr() const;
  float secondary_lr() const;
  float weight_l1() const;
  float bias_l1() const;
  float weight_l2() const;
  float bias_l2() const;
  int32_t DataPartNum() const;
  int32_t loss_scale_factor() const;

 private:
  void Init();

  JobConfigProto job_conf_;
  int64_t job_id_;
};

typedef HashMap<std::string, int64_t> JobName2JobId;

class GlobalJobDescScope final {
 public:
  GlobalJobDescScope(const JobConfigProto& job_conf, int64_t job_id);
  ~GlobalJobDescScope();
};
const JobDesc& GlobalJobDesc();

bool IsPullJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info);
bool IsPushJob(const std::string& job_name, const InterUserJobInfo& inter_user_job_info);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
