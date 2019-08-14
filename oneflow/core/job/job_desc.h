#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {

bool IsInterfaceOpConf(const OperatorConf& op_conf);

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  JobDesc() = default;
  JobDesc(const Job& job, int32_t job_id);
  ~JobDesc() = default;

  // Common
  int32_t job_id() const { return job_id_; }
  const std::string& job_name() const { return job_.job_conf().job_name(); }
  bool is_push_job() const { return is_push_job_; }
  bool is_pull_job() const { return is_pull_job_; }
  const PbRpf<std::string>& arg_op_name() const { return job_.job_conf().arg_op_name(); }
  int64_t concurrency_width() const { return job_.job_conf().concurrency_width(); }
  const JobConfigProto& job_conf() const { return job_.job_conf(); }
  DataType DefaultDataType() const { return job_.job_conf().default_data_type(); }
  size_t SizeOfOneDataId() const { return job_.job_conf().max_data_id_length() * sizeof(char); }
  bool EnableCudnn() const { return job_.job_conf().enable_cudnn(); }
  bool IsTrain() const { return job_.job_conf().has_train_conf(); }
  bool IsPredict() const { return job_.job_conf().has_predict_conf(); }
  int64_t RecordPieceSize() const { return job_.job_conf().piece_size(); }
  int64_t piece_num_of_experiment_phase() const;
  bool enable_experiment_run() const;
  float available_zone_mem_ratio() const;
  size_t persistence_buf_byte() const;
  size_t reserved_host_mem_byte() const;
  size_t reserved_device_mem_byte() const;
  bool save_downloaded_file_to_local_fs() const;
  size_t rdma_mem_block_byte() const;
  size_t rdma_recv_msg_buf_byte() const;
  bool enable_mem_sharing() const { return job_.job_conf().enable_mem_sharing(); }
  bool enable_inplace() const { return job_.job_conf().enable_inplace(); }
  bool enable_true_half_config_when_conv() const {
    return job_.job_conf().enable_true_half_config_when_conv();
  }
  bool enable_float_compute_for_half_gemm() const {
    return job_.job_conf().enable_float_compute_for_half_gemm();
  }
  bool enable_auto_mixed_precision() const { return job_.job_conf().enable_auto_mixed_precision(); }
  const FileSystemConf& data_fs_conf() const;
  const FileSystemConf& snapshot_fs_conf() const;
  bool enable_write_snapshot() const;
  bool write_snapshot_to_master() const { return snapshot_fs_conf().has_localfs_conf(); }
  bool enable_nccl() const { return job_.job_conf().enable_nccl(); }
  bool use_nccl_inter_node_communication() const {
    return job_.job_conf().use_nccl_inter_node_communication();
  }
  int64_t all_reduce_group_num() const;
  int64_t all_reduce_group_min_byte() const;
  float all_reduce_group_size_warmup() const;
  float all_reduce_lazy_ratio() const;
  bool all_reduce_fp16() const;
  int64_t cudnn_buf_limit_mbyte() const { return job_.job_conf().cudnn_buf_limit_mbyte(); }
  bool enable_cuda_ring_all_reduce() const { return job_.job_conf().enable_cuda_ring_all_reduce(); }
  bool cuda_ring_all_reduce_enable_p2p() const {
    return job_.job_conf().cuda_ring_all_reduce_enable_p2p();
  }

  // Train conf
  int32_t NumOfBatchesInSnapshot() const;
  int64_t TotalBatchNum() const;
  int32_t PieceNumOfPrintLoss() const;
  int32_t PieceNumOfPrintAccuracy() const;
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

  Job job_;
  int32_t job_id_;
  bool is_push_job_;
  bool is_pull_job_;
};

typedef HashMap<std::string, int64_t> JobName2JobId;

void WithJobIdGlobal(int64_t job_id, const std::function<void()>& Handler);
const JobDesc& GlobalJobDesc();
const JobDesc& GlobalJobDesc(int64_t job_id);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
