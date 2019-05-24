#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  JobDesc() = default;
  ~JobDesc() = default;

  // Common
  const JobConf& job_conf() const { return job_conf_; }
  const Resource& resource() const { return job_conf_.resource(); }
  const Config& other_conf() const { return job_conf_.other(); }
  const std::string& MdLoadSnapshotPath() { return job_conf_.other().model_load_snapshot_path(); }
  DataType DefaultDataType() const { return job_conf_.other().default_data_type(); }
  size_t SizeOfOneDataId() const { return job_conf_.other().max_data_id_length() * sizeof(char); }
  bool use_rdma() const { return job_conf_.other().use_rdma(); }
  bool EnableCudnn() const { return job_conf_.other().enable_cudnn(); }
  int64_t TotalMachineNum() const { return job_conf_.resource().machine().size(); }
  int32_t CpuDeviceNum() const { return job_conf_.resource().cpu_device_num(); }
  void SetCpuDeviceNum(int32_t val) { job_conf_.mutable_resource()->set_cpu_device_num(val); }
  int32_t GpuDeviceNum() const { return job_conf_.resource().gpu_device_num(); }
  int32_t MemZoneNum() const { return GpuDeviceNum() + 1; }
  int32_t CommNetWorkerNum() const { return job_conf_.resource().comm_net_worker_num(); }
  int32_t MaxMdSaveWorkerNum() const { return job_conf_.resource().max_mdsave_worker_num(); }
  bool IsTrain() const { return job_conf_.other().has_train_conf(); }
  bool IsPredict() const { return job_conf_.other().has_predict_conf(); }
  int64_t RecordPieceSize() const { return job_conf_.other().piece_size(); }
  int64_t piece_num_of_experiment_phase() const;
  bool enable_experiment_run() const;
  float available_zone_mem_ratio() const;
  size_t persistence_buf_byte() const;
  size_t reserved_host_mem_byte() const;
  size_t reserved_device_mem_byte() const;
  bool save_downloaded_file_to_local_fs() const;
  size_t rdma_mem_block_byte() const;
  size_t rdma_recv_msg_buf_byte() const;
  bool collect_act_event() const { return job_conf_.other().collect_act_event(); }
  bool enable_mem_sharing() const { return job_conf_.other().enable_mem_sharing(); }
  const FileSystemConf& data_fs_conf() const;
  const FileSystemConf& snapshot_fs_conf() const;
  bool enable_write_snapshot() const;
  bool write_snapshot_to_master() const { return snapshot_fs_conf().has_localfs_conf(); }
  bool enable_blob_mem_sharing() const { return job_conf_.other().enable_blob_mem_sharing(); }
  bool enable_nccl() const { return job_conf_.other().enable_nccl(); }
  bool use_nccl_inter_node_communication() const {
    return job_conf_.other().use_nccl_inter_node_communication();
  }
  int64_t all_reduce_group_num() const;
  int64_t all_reduce_group_min_byte() const;
  float all_reduce_group_size_warmup() const;
  float all_reduce_lazy_ratio() const;
  bool all_reduce_fp16() const;
  bool use_boxing_v2() const;
  int64_t cudnn_buf_limit_mbyte() const { return job_conf_.other().cudnn_buf_limit_mbyte(); }
  int64_t GetMachineId(const std::string& addr) const;

  // Train conf
  const std::string& MdSaveSnapshotsPath() const;
  int32_t NumOfBatchesInSnapshot() const;
  int64_t TotalBatchNum() const;
  const InitializerConf* DefaultInitializerConf() const;
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

 private:
  friend class Global<JobDesc>;
  JobDesc(const std::string& job_conf_filepath);
  void Init();
  void SanityCheck();
  void SplitDecodeOps();
  void AddRecordLoadOps();

  JobConf job_conf_;
};

const static std::string kProducedLbi2ConsumedDiffLbi = "produced_lbi2consumed_diff_lbi";

std::function<const ParallelConf*(const std::string&)> MakeGetterParallelConf4OpName(
    const Placement& placement);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
