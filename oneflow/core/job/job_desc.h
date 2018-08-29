#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/persistence/file_system.h"

namespace oneflow {

class JobDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobDesc);
  JobDesc() = default;
  ~JobDesc() = default;

  // Common
  const DLNetConf& dlnet_conf() const { return job_conf_.net(); }
  const Resource& resource() const { return job_conf_.resource(); }
  const Placement& placement() const { return job_conf_.placement(); }
  const OtherConf& other_conf() const { return job_conf_.other(); }
  const std::string& MdLoadSnapshotPath() { return job_conf_.other().model_load_snapshot_path(); }
  DataType DefaultDataType() const { return job_conf_.other().default_data_type(); }
  size_t SizeOfOneDataId() const { return job_conf_.other().max_data_id_length() * sizeof(char); }
  bool use_rdma() const { return job_conf_.other().use_rdma(); }
  bool use_synthetic_data() const { return job_conf_.other().use_synthetic_data(); }
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
  int64_t PieceSize() const { return job_conf_.other().piece_size(); }
  int64_t piece_num_of_experiment_phase() const;
  float available_zone_mem_ratio() const;
  size_t persistence_buf_byte() const;
  size_t reserved_host_mem_byte() const;
  size_t reserved_device_mem_byte() const;
  bool save_downloaded_file_to_local_fs() const;
  size_t rdma_mem_block_byte() const;
  size_t rdma_recv_msg_buf_byte() const;
  bool collect_act_event() const { return job_conf_.other().collect_act_event(); }
  bool enable_mem_sharing() const { return job_conf_.other().enable_mem_sharing(); }
  bool enable_write_snapshot() const {
    return IsTrain() && job_conf_.other().enable_write_snapshot();
  }
  bool enable_blob_mem_sharing() const { return job_conf_.other().enable_blob_mem_sharing(); }
  int64_t reduce_group_size() const { return job_conf_.other().reduce_group_size(); }

  // machine_name <-> machine_id
  int64_t MachineID4MachineName(const std::string& machine_name) const;
  const std::string& MachineName4MachineId(int64_t machine_id) const;

  // Train conf
  const std::string& MdSaveSnapshotsPath() const;
  int32_t NumOfBatchesInSnapshot() const;
  int64_t TotalBatchNum() const;
  const InitializerConf* DefaultInitializerConf() const;
  int32_t PieceNumOfPrintLoss() const;
  int32_t PieceNumOfPrintAccuracy() const;
  int64_t BatchSize() const;
  int64_t NumOfPiecesInBatch() const;
  float L1() const;
  float L2() const;
  int32_t DataPartNum() const;

 private:
  friend class Global<JobDesc>;
  JobDesc(const std::string& job_conf_filepath);
  void SplitDecodeOps();
  void AddRecordLoadOps();

  JobConf1 job_conf_;

  HashMap<std::string, int64_t> machine_name2machine_id_;
  HashMap<int64_t, std::string> machine_id2machine_name_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
