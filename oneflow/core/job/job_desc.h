#ifndef ONEFLOW_CORE_JOB_JOB_DESC_H_
#define ONEFLOW_CORE_JOB_JOB_DESC_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/dlnet_conf.pb.h"
#include "oneflow/core/job/job_desc.pb.h"
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
  const JobConf& job_conf() const { return job_conf_; }
  const DLNetConf& dlnet_conf() const { return dlnet_conf_; }
  const Resource& resource() const { return resource_; }
  const Placement& placement() const { return placement_; }
  const std::string& MdLoadSnapshotPath();
  DataType DefaultDataType() const { return job_conf_.default_data_type(); }
  size_t SizeOfOneDataId() const;
  bool use_rdma() const { return job_conf_.use_rdma(); }
  bool UseCudnnOnGpu() const { return job_conf_.use_cudnn_on_gpu(); }
  int64_t TotalMachineNum() const { return resource_.machine().size(); }
  int32_t CpuDeviceNum() const { return resource_.cpu_device_num(); }
  void SetCpuDeviceNum(int32_t val) { resource_.set_cpu_device_num(val); }
  int32_t GpuDeviceNum() const { return resource_.gpu_device_num(); }
  int32_t CommNetWorkerNum() const;
  int32_t PersistenceWorkerNum() const;
  bool IsTrain() const { return job_conf_.has_train_conf(); }
  bool IsPredict() const { return job_conf_.has_predict_conf(); }
  int32_t SinglePieceSize() const { return job_conf_.single_piece_size(); }
  int32_t ParallelPieceSize() const;
  int64_t piece_num_of_experiment_phase() const;
  float available_zone_mem_ratio() const;
  uint64_t persistence_buffer_byte_size() const;
  uint64_t reserved_host_mem_byte_size() const;
  uint64_t reserved_device_mem_byte_size() const;
  bool save_downloaded_file_to_local_fs() const;

  // Train conf
  const std::string& MdSaveSnapshotsPath() const;
  int32_t NumOfBatchesInSnapshot() const;
  int32_t NumOfPiecesInBatch() const;
  int32_t Staleness() const;
  int64_t TotalBatchNum() const;
  const InitializerConf* DefaultInitializerConf() const;
  int32_t PieceNumOfPrintLoss() const;
  int32_t BatchSize() const;
  float L1() const;
  float L2() const;

 private:
  friend class Global<JobDesc>;
  JobDesc(const JobDescProto&);
  void SplitDecodeOps();

  JobConf job_conf_;
  DLNetConf dlnet_conf_;
  Resource resource_;
  Placement placement_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_DESC_H_
